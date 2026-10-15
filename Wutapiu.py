# -*- coding: utf-8 -*-
"""
Wutapiu v6.2 AI
Iandk Labs - DOI: 10.5281/zenodo.23404125 — OCT-2026
====================================
 Zero Mutation on Read
  • learning=False never touches clouds: no empty list creation
  • Clouds keyed by lexeme; synonyms return real surface forms
  • Representatives exclude center word entirely
  • Ingest learns; generation listens purely
V6: Periodic same-sense cloud merging
V6.1: Fixes — resonance, ingests persistence, weaken_link DB
V6.2: Fixes — KeyError protection, use SenseCloud.merge()
Philosophy:
  The crystal does not carve itself while it reflects.
  What has not been learned cannot be assumed to exist.
  Structure holds kinship; context holds meaning.
  Form is the face; the root is the voice.
"""
import re
import math
import json
import hashlib
import logging
from pathlib import Path
from collections import Counter
import numpy as np
# ── Configuration ───────────────────────────────────────────────
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
EPS = 1e-12
BINS = 128
WINDOW = 6
MIN_COST = 0.01
SENSE_THRESHOLD = 0.35
CLOUD_MERGE_THRESHOLD = 0.85
CONTEXT_WEIGHT_DECAY = 0.1
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_REGISTRY = CORPUS_DIR / ".processed.json"
# ── Corpus utilities ───────────────────────────────────────────
def ensure_corpus_dir():
    CORPUS_DIR.mkdir(exist_ok=True)
def load_processed_registry():
    if CORPUS_REGISTRY.exists():
        with open(CORPUS_REGISTRY, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
def save_processed_registry(registry):
    with open(CORPUS_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
def read_pdf_file(file_path):
    try:
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logging.warning("PyPDF2 not installed — pip install PyPDF2")
        return None
    except Exception as e:
        logging.error("Error reading %s: %s", file_path.name, e)
        return None
def scan_corpus():
    ensure_corpus_dir()
    registry = load_processed_registry()
    new_files = []
    for ext in ["*.txt", "*.pdf"]:
        for f in CORPUS_DIR.glob(ext):
            file_hash = get_file_hash(f)
            if str(f) not in registry or registry[str(f)] != file_hash:
                new_files.append((f, file_hash))
    return new_files
# ============================================================
# PART 1: SPARSE STRUCTURES
# ============================================================
class SparseBinaryMatrix:
    def __init__(self, name=""):
        self.name = name
        self._rows = {}
        self._row_index = {}
        self._col_index = {}
        self._row_label = []
        self._col_label = []
    def add_row(self, label):
        if label not in self._row_index:
            idx = len(self._row_label)
            self._row_index[label] = idx
            self._row_label.append(label)
            self._rows[idx] = set()
        return self._row_index[label]
    def add_col(self, label):
        if label not in self._col_index:
            idx = len(self._col_label)
            self._col_index[label] = idx
            self._col_label.append(label)
        return self._col_index[label]
    def set(self, row_label, col_label, value=True):
        r = self.add_row(row_label)
        c = self.add_col(col_label)
        if value:
            self._rows[r].add(c)
        else:
            self._rows[r].discard(c)
    def get(self, row_label, col_label):
        r = self._row_index.get(row_label)
        c = self._col_index.get(col_label)
        if r is None or c is None:
            return False
        return c in self._rows.get(r, set())
    def row(self, label):
        r = self._row_index.get(label)
        if r is None:
            return set()
        return {self._col_label[c] for c in self._rows.get(r, set())}
    def row_labels(self):
        return list(self._row_label)
    def similarity_jaccard(self, a, b):
        sa, sb = self.row(a), self.row(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
    def complementarity(self, a, b):
        sa, sb = self.row(a), self.row(b)
        union = sa | sb
        if not union:
            return 0.0
        return len((sa - sb) | (sb - sa)) / len(union)
    def dissimilarity_jaccard(self, a, b):
        return self.complementarity(a, b)
    def export_rows(self):
        return [
            (r, lbl, sorted([self._col_label[c_idx] for c_idx in self._rows.get(r, set())]))
            for r, lbl in enumerate(self._row_label)
        ]
    def import_rows(self, data):
        for r, lbl, cols_ref in data:
            self.add_row(lbl)
            for col_ref in cols_ref:
                col_label = col_ref if isinstance(col_ref, str) else \
                            (self._col_label[col_ref] if 0 <= col_ref < len(self._col_label) else None)
                if col_label:
                    self.set(lbl, col_label, True)
class MarkovRelationalMatrix:
    def __init__(self):
        self._states = {}
        self._state_list = []
        self._trans = {}
    def add_state(self, label):
        if label not in self._states:
            self._states[label] = len(self._state_list)
            self._state_list.append(label)
            self._trans[label] = set()
    def set_transition(self, a, b):
        self.add_state(a)
        self.add_state(b)
        self._trans[a].add(b)
    def get_transitions(self, label):
        return self._trans.get(label, set())
    def export_all(self):
        return [(a, sorted(bset)) for a, bset in self._trans.items()]
    def import_trans(self, data):
        for a, targets in data:
            for b in targets:
                self.set_transition(a, b)
# ============================================================
# PART 2: LANGUAGE PROFILES
# ============================================================
class LanguageProfile:
    code = ""
    name = ""
    def morphemes(self): raise NotImplementedError
    def stem_candidates(self, stem): raise NotImplementedError
    def tokenize_sentences(self, text): raise NotImplementedError
class SpanishProfile(LanguageProfile):
    code = "es"
    name = "spanish"
    _MORPH = {
        "PLURAL": [("s", "suffix"), ("es", "suffix")],
        "FEMININE": [("a", "suffix")],
        "MASCULINE": [("o", "suffix")],
        "DIMINUTIVE_ITO": [("ito", "suffix"), ("ita", "suffix")],
        "INFINITIVE": [("ar", "suffix"), ("er", "suffix"), ("ir", "suffix")],
        "GERUND": [("ando", "suffix"), ("iendo", "suffix")],
        "PARTICIPLE": [("ado", "suffix"), ("ido", "suffix")],
        "FUTURE": [("ré", "suffix"), ("rás", "suffix"), ("rá", "suffix")],
        "NEGATION_NO": [("no", "prefix")],
    }
    def morphemes(self): return self._MORPH
    def stem_candidates(self, stem):
        out = {stem}
        if stem.endswith("qu"): out.add(stem[:-2] + "c")
        if stem.endswith("gu"): out.add(stem[:-2] + "g")
        for ac, pl in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u")]:
            if ac in stem: out.add(stem.replace(ac, pl))
        return out
    def tokenize_sentences(self, text):
        sentences = re.split(r"[.!?]+", text)
        return [re.findall(r"[A-Za-záéíóúñüÁÉÍÓÚÑÜ0-9]+", s) for s in sentences if s.strip()]
class EnglishProfile(LanguageProfile):
    code = "en"
    name = "english"
    _MORPH = {
        "PLURAL": [("s", "suffix"), ("es", "suffix")],
        "3RD_PERSON_SINGULAR": [("s", "suffix")],
        "PAST_TENSE": [("ed", "suffix")],
        "GERUND": [("ing", "suffix")],
        "PREFIX_UN": [("un", "prefix")],
        "ADVERB_LY": [("ly", "suffix")],
    }
    def morphemes(self): return self._MORPH
    def stem_candidates(self, stem):
        out = {stem, stem + "e"}
        if len(stem) >= 2 and stem[-1] == stem[-2]: out.add(stem[:-1])
        return out
    def tokenize_sentences(self, text):
        sentences = re.split(r"[.!?]+", text)
        return [re.findall(r"[A-Za-z0-9]+", s) for s in sentences if s.strip()]
PROFILES = {"es": SpanishProfile, "en": EnglishProfile}
# ============================================================
# PART 3: SENSE CLOUDS — PURE READ MODE
# ============================================================
class SenseCloud:
    __slots__ = ["context_sum", "context_mean", "mass", "representatives"]
    def __init__(self):
        self.context_sum = np.zeros(BINS, complex)
        self.context_mean = np.zeros(BINS, complex)
        self.mass = 0.0
        self.representatives = Counter()
    def update(self, context_wave, context_words, weight=1.0):
        self.context_sum += context_wave * weight
        self.mass += weight
        norm = np.linalg.norm(self.context_sum)
        self.context_mean = self.context_sum / norm if norm >= EPS else self.context_sum
        for w in context_words:
            self.representatives[w] += weight
    def resonance(self, spectral_core, context_wave):
        if self.mass < EPS:
            return 0.0
        return spectral_core._resonance(self.context_mean, context_wave)
    def merge(self, other, spectral_core):
        self.context_sum += other.context_sum
        self.mass += other.mass
        norm = np.linalg.norm(self.context_sum)
        self.context_mean = self.context_sum / norm if norm >= EPS else self.context_sum
        self.representatives.update(other.representatives)
    def to_dict(self):
        return {
            "context_sum_real": [float(x) for x in np.real(self.context_sum)],
            "context_sum_imag": [float(x) for x in np.imag(self.context_sum)],
            "mass": self.mass,
            "representatives": dict(self.representatives),
        }
    @classmethod
    def from_dict(cls, data):
        cloud = cls()
        cr = np.array(data["context_sum_real"], dtype=np.float64)
        ci = np.array(data["context_sum_imag"], dtype=np.float64)
        cloud.context_sum = cr + 1j * ci
        norm = np.linalg.norm(cloud.context_sum)
        cloud.context_mean = cloud.context_sum / norm if norm >= EPS else cloud.context_sum
        cloud.mass = data["mass"]
        cloud.representatives = Counter(data["representatives"])
        return cloud
class SpectralSenseClouds:
    """Clouds keyed by lexeme — read mode = zero mutation"""
    def __init__(self, spectral_core):
        self.spectral = spectral_core
        self.clouds = {}
        self.lexemes = self.clouds

    def merge_clouds(self, threshold=0.80, min_mass=1.0, verbose=True):
        """
        V6: Fusiona nubes del MISMO sentido que nacieron separadas.
        Se llama cada 100 ingests. No toca sentidos distintos.
        V6.1: Usa resonance espectral en lugar de coseno
        V6.2: Reutiliza SenseCloud.merge()
        """
        total_merged = 0
        for lexema, nubes in self.lexemes.items():
            if len(nubes) < 2:
                continue
            merged_this_lexeme = True
            while merged_this_lexeme and len(nubes) >= 2:
                merged_this_lexeme = False
                best_pair = None
                best_sim = threshold
                for i in range(len(nubes)):
                    for j in range(i + 1, len(nubes)):
                        a, b = nubes[i], nubes[j]
                        if a.mass < min_mass or b.mass < min_mass:
                            continue
                        sim = self.spectral._resonance(a.context_mean, b.context_mean)
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (i, j)
                if best_pair is not None:
                    i, j = best_pair
                    a, b = nubes[i], nubes[j]
                    if a.mass >= b.mass:
                        keep, drop = a, b
                    else:
                        keep, drop = b, a
                    # PATCH v6.2: Usa el método merge de la propia nube
                    keep.merge(drop, self.spectral)
                    nubes.remove(drop)
                    total_merged += 1
                    merged_this_lexeme = True
                    if verbose:
                        print(f"[MERGE] '{lexema}': fusionadas 2 nubes sim={best_sim:.3f} -> masa nueva {keep.mass:.1f}")
        return total_merged

    def _make_context_wave(self, words, center_idx):
        wave = np.zeros(BINS, complex)
        total_weight = 0.0
        for j, w in enumerate(words):
            if j == center_idx:
                continue
            if w not in self.spectral.nodes:
                continue
            dist = abs(j - center_idx)
            weight = math.exp(-CONTEXT_WEIGHT_DECAY * dist)
            wave += self.spectral._word_wave(w) * weight
            total_weight += weight
        if total_weight < EPS:
            return np.zeros(BINS, complex)
        norm = np.linalg.norm(wave)
        return wave / norm if norm >= EPS else wave
    def assign_sense(self, lexeme, sentence_words, center_idx, learning: bool = True):
        if lexeme not in self.clouds:
            if not learning:
                return 0, 0.0
            self.clouds[lexeme] = []
        context_wave = self._make_context_wave(sentence_words, center_idx)
        context_words_no_center = [w for j, w in enumerate(sentence_words) if j != center_idx]
        if np.linalg.norm(context_wave) < EPS:
            return 0, 0.0
        clouds = self.clouds[lexeme]
        if not clouds:
            if learning:
                clouds.append(SenseCloud())
                clouds[-1].update(context_wave, context_words_no_center, weight=1.0)
            return 0, 1.0 if learning else 0.0
        scores = [c.resonance(self.spectral, context_wave) for c in clouds]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        if not learning:
            return best_idx, best_score
        if best_score >= SENSE_THRESHOLD:
            clouds[best_idx].update(context_wave, context_words_no_center, weight=best_score)
            to_merge = []
            for i in range(len(clouds)):
                if i == best_idx:
                    continue
                sim = self.spectral._resonance(clouds[best_idx].context_mean, clouds[i].context_mean)
                if sim >= CLOUD_MERGE_THRESHOLD:
                    to_merge.append(i)
            for i in reversed(to_merge):
                clouds[best_idx].merge(clouds[i], self.spectral)
                clouds.pop(i)
            return best_idx, best_score
        else:
            clouds.append(SenseCloud())
            clouds[-1].update(context_wave, context_words_no_center, weight=1.0)
            return len(clouds) - 1, best_score
    def get_context_identity(self, lexeme, surface_token, sentence_words, center_idx, learning: bool = True):
        if surface_token not in self.spectral.nodes:
            return None, 0.0
        base_wave = self.spectral.get_identity(surface_token)
        cloud_idx, score = self.assign_sense(lexeme, sentence_words, center_idx, learning=learning)
        if lexeme not in self.clouds or not self.clouds[lexeme]:
            return base_wave, 0.0
        cloud = self.clouds[lexeme][cloud_idx]
        ctx_comp = self.spectral._normalize(cloud.context_mean)
        modulated = self.spectral._normalize(base_wave + ctx_comp * score)
        return modulated, score
    def cloud_resonance_between(self, lexeme_a, lexeme_b):
        if lexeme_a not in self.clouds or lexeme_b not in self.clouds:
            return 0.0
        max_res = 0.0
        for ca in self.clouds[lexeme_a]:
            for cb in self.clouds[lexeme_b]:
                res = self.spectral._resonance(ca.context_mean, cb.context_mean)
                if res > max_res:
                    max_res = res
        return max_res
    def get_cloud_summary(self, lexeme):
        if lexeme not in self.clouds:
            return []
        return [
            {
                "cloud_index": i,
                "mass": c.mass,
                "top_context": c.representatives.most_common(5),
            }
            for i, c in enumerate(self.clouds[lexeme])
        ]
    def to_dict(self):
        return {lex: [c.to_dict() for c in clist] for lex, clist in self.clouds.items()}
    def load_from_dict(self, data):
        for lex, clist in data.items():
            self.clouds[lex] = [SenseCloud.from_dict(cd) for cd in clist]
# ============================================================
# PART 4: SPECTRAL CORE
# ============================================================
class SpectralCore:
    def __init__(self):
        self.nodes = {}
        self.links = {}
        self._wave_cache = {}
        self._vocab_mean = None
        self._dirty_nodes = set()
        self._dirty_links = set()
    def _normalize(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        norm = np.linalg.norm(spectrum)
        return spectrum / norm if norm >= EPS else np.zeros(BINS, complex)
    def _word_wave(self, word):
        if word in self._wave_cache:
            return self._wave_cache[word]
        k = np.arange(BINS, dtype=float)
        wave = np.zeros(BINS, complex)
        max_freq = max(1, BINS // 4 - 1)
        for i, char in enumerate(word):
            freq = (ord(char) % max_freq) + 1
            phase = self._hash_float("ph", word, i, char) * 2 * math.pi
            wave += np.exp(1j * ((2 * math.pi * freq * k / BINS) + phase))
        wave = self._normalize(wave / len(word))
        self._wave_cache[word] = wave
        return wave
    def _hash_float(self, *parts):
        data = "|".join(repr(p) for p in parts).encode()
        return int(hashlib.sha3_256(data).hexdigest()[:8], 16) / 2**32
    def _resonance(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < EPS or nb < EPS:
            return 0.0
        return max(0.0, min(1.0, abs(np.sum(a * np.conj(b))) / (na * nb)))
    def _interference(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < EPS or nb < EPS:
            return 0.0
        return max(-1.0, min(1.0, np.real(np.sum(a * np.conj(b))) / (na * nb)))
    def _phase(self, spectrum):
        w = abs(spectrum)
        tw = w.sum()
        if tw < EPS:
            return 0.0
        vec = np.sum(w * np.exp(1j * np.angle(spectrum)))
        return ((np.angle(vec) % (2 * math.pi)) / (2 * math.pi)) if abs(vec) >= EPS else 0.0
    def _spectrum_xyz(self, spectrum):
        mag = abs(spectrum)
        total = mag.sum()
        if total < EPS:
            return np.zeros(3)
        k = np.arange(BINS, dtype=float)
        x = (k * mag).sum() / total / BINS
        y = float(np.argmax(mag)) / BINS
        p = mag / total
        z = (-np.sum(p * np.log2(p + EPS))) / math.log2(BINS)
        return np.array([x, y, z])
    def learn_token(self, token):
        wave = self._word_wave(token)
        if token in self.nodes:
            d = self.nodes[token]
            d["count"] += 1
            d["sum"] += wave
            d["spectrum"] = self._normalize(d["sum"])
            d["mass"] = 1 + math.log1p(d["count"])
            d["xyz"] = self._spectrum_xyz(d["spectrum"])
            d["phase"] = self._phase(d["spectrum"])
        else:
            normed = self._normalize(wave)
            self.nodes[token] = {
                "sum": wave.copy(), "spectrum": normed, "count": 1, "mass": 1.0,
                "xyz": self._spectrum_xyz(normed), "phase": self._phase(normed)
            }
        self._dirty_nodes.add(token)
        self._vocab_mean = None
    def link_tokens(self, a, b, weight=1.0):
        if a not in self.nodes or b not in self.nodes:
            return
        key = tuple(sorted((a, b)))
        self._dirty_links.add(key)
        combined = self._normalize(self.nodes[a]["spectrum"] + self.nodes[b]["spectrum"])
        if key in self.links:
            self.links[key]["sum"] += combined * weight
            self.links[key]["count"] += weight
            self.links[key]["spectrum"] = self._normalize(self.links[key]["sum"])
        else:
            self.links[key] = {"sum": combined * weight, "spectrum": combined, "count": weight}
    def weaken_link(self, a, b, weight=1.0):
        # PATCH v6.2: protección contra KeyError por nodos desaparecidos
        key = tuple(sorted((a, b)))
        if key not in self.links:
            return
        # Si los nodos ya no existen, no se puede recalcular: borrar el enlace huérfano
        if a not in self.nodes or b not in self.nodes:
            self.links.pop(key, None)
            self._dirty_links.add(key)
            return
        link = self.links[key]
        combined = self._normalize(self.nodes[a]["spectrum"] + self.nodes[b]["spectrum"])
        link["count"] = max(0.0, link["count"] - weight)
        link["sum"] -= combined * weight
        link["spectrum"] = self._normalize(link["sum"])
        if link["count"] <= EPS:
            self.links.pop(key, None)
            self._dirty_links.add(key)
        else:
            self._dirty_links.add(key)
    def get_vocabulary_mean(self):
        if self._vocab_mean is not None:
            return self._vocab_mean
        if not self.nodes:
            return np.zeros(BINS, complex)
        self._vocab_mean = np.mean([d["spectrum"] for d in self.nodes.values()], axis=0)
        return self._vocab_mean
    def get_identity(self, word):
        mean = self.get_vocabulary_mean()
        spec = self.nodes[word]["spectrum"]
        identity = self._normalize(spec - mean)
        return identity if np.linalg.norm(identity) >= EPS else self._normalize(spec)
    def build_adjacency(self):
        adj = {}
        for (a, b), link in self.links.items():
            s = math.log1p(max(0.0, link["count"]))
            adj.setdefault(a, []).append((b, s))
            adj.setdefault(b, []).append((a, s))
        return adj
# ============================================================
# PART 5: INTEGRATED CORE
# ============================================================
class CheyenneLakota:
    def __init__(self, language="es"):
        self.language_code = language
        self.db_path = f"wutapiu_{language}.db"
        self.profile = PROFILES.get(language, SpanishProfile)()
        self.spectral = SpectralCore()
        self.sense_clouds = SpectralSenseClouds(self.spectral)
        self.M_plus = SparseBinaryMatrix("M_plus")
        self.T_markov = MarkovRelationalMatrix()
        self.lexeme_frequency = {}
        self.hierarchy_edges = []
        self.token_to_lexeme = {}
        self.lexeme_to_forms = {}
        self.connection = None
        self.ingests_total = 0
        try:
            import sqlite3
            self.connection = sqlite3.connect(self.db_path)
        except ImportError:
            pass
        self._init_db()
        self._load()
        self._start_corpus()
    def close(self):
        self._save()
        if self.connection:
            self.connection.commit()
            self.connection.close()
        print("Saved. Goodbye.")
    def _init_db(self):
        if not self.connection:
            return
        cur = self.connection.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS spectral_nodes(
            word TEXT PRIMARY KEY, sum_blob BLOB, count INTEGER, mass REAL
        );
        CREATE TABLE IF NOT EXISTS spectral_links(
            w1 TEXT, w2 TEXT, sum_blob BLOB, count REAL, PRIMARY KEY(w1, w2)
        );
        CREATE TABLE IF NOT EXISTS m_plus_rows(row_label TEXT, col_label TEXT, PRIMARY KEY(row_label, col_label));
        CREATE TABLE IF NOT EXISTS markov_transitions(from_label TEXT, to_label TEXT);
        CREATE TABLE IF NOT EXISTS extra_data(key TEXT PRIMARY KEY, value TEXT);
        """)
        self.connection.commit()
    def _start_corpus(self):
        new = scan_corpus()
        if not new:
            return
        print(f" {len(new)} new corpus file(s)")
        reg = load_processed_registry()
        for path, h in new:
            print(f"   → {path.name}")
            text = read_text_file(path) if path.suffix.lower() == ".txt" else read_pdf_file(path)
            if text and text.strip():
                n = self.ingest(text)
                print(f"    {n} sentences")
            reg[str(path)] = h
        save_processed_registry(reg)
        print()
    def _extract_stem(self, word_form):
        if word_form in self.token_to_lexeme:
            return self.token_to_lexeme[word_form]
        morph = self.profile.morphemes()
        candidates = {word_form}
        def expand(curr, depth=0):
            candidates.add(curr)
            if depth >= 3:
                return
            for cat, vars in morph.items():
                for m, pos in vars:
                    if pos == "prefix" and curr.startswith(m) and len(curr) - len(m) >= 2:
                        expand(curr[len(m):], depth + 1)
            for cat, vars in morph.items():
                for m, pos in vars:
                    if pos == "suffix" and curr.endswith(m) and len(curr) - len(m) >= 2:
                        expand(curr[:-len(m)], depth + 1)
        expand(word_form)
        valid = [c for c in candidates if len(c) >= 2]
        if not valid:
            lex = word_form
        else:
            valid.sort(key=lambda x: (self.lexeme_frequency.get(x, 0), len(x)), reverse=True)
            lex = valid[0]
        self.token_to_lexeme[word_form] = lex
        return lex
    def _register_form_for_lexeme(self, lexeme, surface_form):
        if lexeme not in self.lexeme_to_forms:
            self.lexeme_to_forms[lexeme] = Counter()
        self.lexeme_to_forms[lexeme][surface_form] += 1
    def _get_best_form_for_lexeme(self, lexeme):
        if lexeme in self.lexeme_to_forms and self.lexeme_to_forms[lexeme]:
            return self.lexeme_to_forms[lexeme].most_common(1)[0][0]
        return lexeme
    def _reg_morphology(self, lexeme, form):
        morph = self.profile.morphemes()
        pre, suf = [], []
        rem = form
        for cat, vars in morph.items():
            for m, pos in vars:
                if pos == "prefix" and rem.startswith(m):
                    pre.append(cat); rem = rem[len(m):]
        temp = form
        for cat, vars in morph.items():
            for m, pos in vars:
                if pos == "suffix" and temp.endswith(m):
                    suf.append(cat); temp = temp[:-len(m)]
        active = pre + suf
        for c in active:
            self.M_plus.set(lexeme, c)
        for i in range(len(active) - 1):
            self.T_markov.set_transition(active[i], active[i+1])
    def _hierarchy_scan(self, tokens):
        txt = " ".join(tokens)
        for pat in [r"(\w+)\s+(?:is|are|es|son)\s+(?:(?:a|an|the|un|una|los|las)\s+)?(\w+)"]:
            for m in re.finditer(pat, txt):
                la = self._extract_stem(m.group(1))
                lb = self._extract_stem(m.group(2))
                if la != lb and (la, lb) not in self.hierarchy_edges:
                    self.hierarchy_edges.append((la, lb))
    def ingest(self, text):
        sentences = self.profile.tokenize_sentences(text)
        for sent in sentences:
            for idx, tok in enumerate(sent):
                lex = self._extract_stem(tok)
                self.lexeme_frequency[lex] = self.lexeme_frequency.get(lex, 0) + 1
                self._register_form_for_lexeme(lex, tok)
                self._reg_morphology(lex, tok)
                self.spectral.learn_token(tok)
                self.sense_clouds.assign_sense(lex, sent, idx, learning=True)
            L = len(sent)
            for i in range(L):
                for j in range(i+1, min(L, i+WINDOW)):
                    self.spectral.link_tokens(sent[i], sent[j], 1.0/(j-i))
            self._hierarchy_scan(sent)
        self.ingests_total += 1
        if self.ingests_total % 100 == 0:
            merged = self.sense_clouds.merge_clouds(threshold=0.80)
            if merged > 0:
                print(f"[V6] Ingest {self.ingests_total}: {merged} fusiones realizadas")
        return len(sentences)
    def find_synonyms(self, lexeme, threshold=0.35, max_candidates=500):
        candidates = []
        for cand_lex in self.M_plus.row_labels():
            if cand_lex == lexeme:
                continue
            struct_sim = self.M_plus.similarity_jaccard(lexeme, cand_lex)
            cloud_res = self.sense_clouds.cloud_resonance_between(lexeme, cand_lex)
            combined = (struct_sim + cloud_res) / 2.0
            if combined >= threshold:
                surface_form = self._get_best_form_for_lexeme(cand_lex)
                candidates.append((surface_form, combined))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:max_candidates]
    def find_antonyms(self, lexeme, threshold=0.6, max_candidates=500):
        cands = []
        for cand_lex in self.M_plus.row_labels():
            if cand_lex == lexeme:
                continue
            dissim = self.M_plus.dissimilarity_jaccard(lexeme, cand_lex)
            if dissim >= threshold:
                surface_form = self._get_best_form_for_lexeme(cand_lex)
                cands.append((surface_form, dissim))
        cands.sort(key=lambda x: -x[1])
        return cands[:max_candidates]
    def generate_response(self, prompt):
        tokens = re.findall(r"[\w]+", prompt)
        if not tokens and not self.spectral.nodes:
            return "I am ready. Tell me anything." if self.language_code == "en" else "Estoy listo. Dime lo que quieras."
        input_wave = np.zeros(BINS, complex)
        for idx, tok in enumerate(tokens):
            if tok not in self.spectral.nodes:
                continue
            lex = self._extract_stem(tok)
            mod_id, _ = self.sense_clouds.get_context_identity(lex, tok, tokens, idx, learning=False)
            if mod_id is None:
                mod_id = self.spectral.get_identity(tok)
            phase = np.exp(1j * 2 * math.pi * idx / max(len(tokens), 1))
            input_wave += mod_id * phase
        input_wave = self.spectral._normalize(input_wave / max(len(tokens), 1) if tokens else input_wave)
        vocab_mean = self.spectral.get_vocabulary_mean()
        input_id = self.spectral._normalize(input_wave - vocab_mean)
        if np.linalg.norm(input_id) < EPS:
            input_id = input_wave
        adj = self.spectral.build_adjacency()
        primary = {t: 1.0 for t in tokens if t in self.spectral.nodes}
        if not primary:
            return max(self.spectral.nodes.items(), key=lambda x: x[1]["mass"])[0]
        expanded = dict(primary)
        for tok in list(primary):
            lex = self._extract_stem(tok)
            for syn_form, score in self.find_synonyms(lex)[:2]:
                if syn_form not in expanded:
                    expanded[syn_form] = score * 0.8
        candidates = dict(expanded)
        for base_form in list(expanded):
            for nb_form, strength in adj.get(base_form, []):
                if nb_form in candidates:
                    continue
                res = self.spectral._resonance(
                    self.spectral.get_identity(nb_form),
                    self.spectral.get_identity(base_form)
                )
                val = res * strength
                nb_lex = self._extract_stem(nb_form)
                base_lex = self._extract_stem(base_form)
                for ant_form, _ in self.find_antonyms(base_lex):
                    if nb_form == ant_form:
                        val *= 0.1
                candidates[nb_form] = max(candidates.get(nb_form, 0), val / (1 + val))
        filtered = {}
        for w, val in candidates.items():
            if self.spectral._interference(self.spectral.get_identity(w), input_id) > 0:
                filtered[w] = val
        if not filtered:
            filtered = candidates
        if not filtered:
            return "I am still listening..." if self.language_code == "en" else "Sigo escuchando..."
        filtered = dict(sorted(filtered.items(), key=lambda x: -x[1])[:100])
        energy = math.log1p(len(tokens)) * (1 + max(0, sum(filtered.values()) / max(len(filtered), 1)))
        current = max(filtered, key=filtered.get)
        traj_id = self.spectral.get_identity(current)
        traj_xyz = self.spectral.nodes[current]["xyz"].copy()
        traj_phase = self.spectral.nodes[current]["phase"]
        result = [current]
        used = {current}
        while energy > EPS and len(result) < 12:
            scores, words = [], []
            for w in filtered:
                if w in used:
                    continue
                nd = self.spectral.nodes[w]
                res = self.spectral._resonance(self.spectral.get_identity(w), traj_id)
                score = filtered[w] * res * math.log1p(nd["mass"])
                w_lex = self._extract_stem(w)
                curr_lex = self._extract_stem(current)
                if (curr_lex, w_lex) in self.hierarchy_edges:
                    score *= 1.5
                scores.append(score)
                words.append(w)
            if not scores:
                break
            total = sum(scores)
            if total < EPS:
                break
            seed = int(hashlib.sha3_256(f"{current}{len(result)}{energy:.15g}".encode()).hexdigest()[:4], 16) / 65535
            cum, sel = 0.0, 0
            for i, s in enumerate(scores):
                cum += s / total
                if cum > seed:
                    sel = i; break
            next_w = words[sel]
            nd = self.spectral.nodes[next_w]
            coh = math.cos(2 * math.pi * abs(traj_phase - nd["phase"]))
            dist = np.linalg.norm(traj_xyz - nd["xyz"]) + EPS
            cost = dist / (1 + math.log1p(nd["mass"])) * max(EPS, 1 - coh)
            energy -= max(MIN_COST, cost)
            traj_id = self.spectral._normalize(traj_id + self.spectral.get_identity(next_w))
            traj_xyz = (traj_xyz * len(result) + nd["xyz"]) / (len(result) + 1)
            traj_phase = (traj_phase + nd["phase"]) % 1.0
            result.append(next_w)
            used.add(next_w)
            current = next_w
        return " ".join(result)
    def correct_response(self, user_input, correction):
        self.ingest(correction)
        itok = re.findall(r"\w+", user_input)
        ctok = re.findall(r"\w+", correction)
        for a in itok:
            for b in ctok:
                if a != b:
                    self.spectral.link_tokens(a, b, 2.0)
        prev = self.generate_response(user_input)
        ptok = re.findall(r"\w+", prev)
        cset = set(ctok)
        for a in itok:
            for b in ptok:
                if b not in cset and a != b:
                    self.spectral.weaken_link(a, b, 1.0)
        return "Learned." if self.language_code == "en" else "Integrado."
    def _save_extra(self, cur):
        extra = {
            "lexeme_frequency": self.lexeme_frequency,
            "hierarchy_edges": self.hierarchy_edges,
            "sense_clouds": self.sense_clouds.to_dict(),
            "token_to_lexeme": self.token_to_lexeme,
            "lexeme_to_forms": {lex: dict(cnt) for lex, cnt in self.lexeme_to_forms.items()},
            "ingests_total": self.ingests_total,
        }
        cur.execute("DELETE FROM extra_data")
        for k, v in extra.items():
            cur.execute("INSERT OR REPLACE INTO extra_data VALUES(?, ?)",
                       (k, json.dumps(v, ensure_ascii=False)))
    def _load_extra(self, cur):
        cur.execute("SELECT key, value FROM extra_data")
        data = {}
        for k, v in cur.fetchall():
            try: data[k] = json.loads(v)
            except: pass
        self.lexeme_frequency = data.get("lexeme_frequency", {})
        self.hierarchy_edges = data.get("hierarchy_edges", [])
        clouds_data = data.get("sense_clouds", {})
        if clouds_data:
            self.sense_clouds.load_from_dict(clouds_data)
        self.token_to_lexeme = data.get("token_to_lexeme", {})
        raw_l2f = data.get("lexeme_to_forms", {})
        self.lexeme_to_forms = {lex: Counter(cnt) for lex, cnt in raw_l2f.items()}
        self.ingests_total = data.get("ingests_total", 0)
    def _save(self):
        if not self.connection:
            return
        cur = self.connection.cursor()
        for w in list(self.spectral._dirty_nodes):
            if w not in self.spectral.nodes:
                continue
            n = self.spectral.nodes[w]
            blob = np.concatenate([np.real(n["sum"]).astype(">f8"), np.imag(n["sum"]).astype(">f8")]).tobytes()
            cur.execute("INSERT OR REPLACE INTO spectral_nodes VALUES(?, ?, ?, ?)",
                       (w, blob, n["count"], n["mass"]))
        self.spectral._dirty_nodes.clear()
        for key in list(self.spectral._dirty_links):
            if key not in self.spectral.links:
                w1, w2 = key
                cur.execute("DELETE FROM spectral_links WHERE w1=? AND w2=?", (w1, w2))
                continue
            w1, w2 = key
            lnk = self.spectral.links[key]
            blob = np.concatenate([np.real(lnk["sum"]).astype(">f8"), np.imag(lnk["sum"]).astype(">f8")]).tobytes()
            cur.execute("INSERT OR REPLACE INTO spectral_links VALUES(?, ?, ?, ?)",
                       (w1, w2, blob, lnk["count"]))
        self.spectral._dirty_links.clear()
        cur.execute("DELETE FROM m_plus_rows")
        for _, lbl, cols in self.M_plus.export_rows():
            for c in cols:
                cur.execute("INSERT INTO m_plus_rows VALUES(?, ?)", (lbl, c))
        cur.execute("DELETE FROM markov_transitions")
        for src, tgts in self.T_markov.export_all():
            for t in tgts:
                cur.execute("INSERT INTO markov_transitions VALUES(?, ?)", (src, t))
        self._save_extra(cur)
        self.connection.commit()
    def _load(self):
        if not self.connection:
            return
        cur = self.connection.cursor()
        cur.execute("SELECT word, sum_blob, count, mass FROM spectral_nodes")
        for w, blob, cnt, m in cur:
            arr = np.frombuffer(blob, dtype=">f8")
            mid = len(arr) // 2
            spec = arr[:mid] + 1j * arr[mid:]
            normed = self.spectral._normalize(spec)
            self.spectral.nodes[w] = {
                "sum": spec, "spectrum": normed, "count": cnt, "mass": m,
                "xyz": self.spectral._spectrum_xyz(normed), "phase": self.spectral._phase(normed)
            }
        cur.execute("SELECT w1, w2, sum_blob, count FROM spectral_links")
        for w1, w2, blob, cnt in cur:
            arr = np.frombuffer(blob, dtype=">f8")
            mid = len(arr) // 2
            spec = arr[:mid] + 1j * arr[mid:]
            self.spectral.links[tuple(sorted((w1, w2)))] = {
                "sum": spec, "spectrum": self.spectral._normalize(spec), "count": cnt
            }
        cur.execute("SELECT row_label, col_label FROM m_plus_rows")
        for r, c in cur:
            self.M_plus.set(r, c, True)
        cur.execute("SELECT from_label, to_label FROM markov_transitions")
        for a, b in cur:
            self.T_markov.set_transition(a, b)
        self._load_extra(cur)
        self.spectral._wave_cache.clear()
def main():
    import sys
    lang = sys.argv[1].lower() if len(sys.argv) > 1 else "es"
    if lang not in PROFILES:
        print(f"Unknown language '{lang}' — using Spanish (es)")
        lang = "es"
    core = CheyenneLakota(lang)
    print(" Wutapiu v6.2 ")
    print(" Learning only on ingest/correction — periodic cloud merging active\n")
    print("Say anything. >>corrige: ... to teach. salir to exit.\n" if lang == "es" else
          "Say anything. >>correct: ... to teach. quit to exit.\n")
    last_input = ""
    try:
        while True:
            inp = input("> ").strip()
            if not inp:
                continue
            if inp.lower() in {"quit", "exit", "salir"}:
                break
            if inp.startswith(">>correct:") or inp.startswith(">>corrige:"):
                corr = inp.split(":", 1)[1].strip()
                print(f"◦ {core.correct_response(last_input, corr)}\n")
                continue
            last_input = inp
            core.ingest(inp)
            reply = core.generate_response(inp)
            print(f"◦ {reply}\n")
    except KeyboardInterrupt:
        print()
    finally:
        core.close()
if __name__ == "__main__":
    main()
