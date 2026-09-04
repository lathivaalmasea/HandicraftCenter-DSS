import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import base64
import io

st.set_page_config(page_title="SPK Handicraft Tourism", layout="wide")

# BACKGROUND & CSS
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(8,18,38,0.68), rgba(8,18,38,0.68)),
                    url("data:image/jpeg;base64,{img_data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    h1,h2,h3,h4,h5,h6,p,span,label,li,.stMarkdown {{ color: white !important; }}
    .info-box {{
        background: rgba(10,25,55,0.82);
        border: 1px solid rgba(255,210,80,0.35);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }}
    .info-box h4, .info-box strong, .info-box b {{ color: #FFD580 !important; }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(10,25,55,0.75);
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(255,190,50,0.92) !important;
        color: black !important;
    }}
    div[data-testid="metric-container"] {{
        background: rgba(10,25,55,0.82);
        border: 1px solid rgba(255,210,80,0.35);
        border-radius: 10px;
        padding: 10px;
    }}
    .stButton > button {{
        background: rgba(255,185,40,0.92);
        color: black;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 24px;
    }}
    .stButton > button:hover {{ background: rgba(255,210,70,1); }}
    .stDataFrame, div[data-testid="stExpander"] {{
        background: rgba(10,25,55,0.80);
        border-radius: 8px;
    }}
    hr {{ border-color: rgba(255,210,80,0.30); }}
    .step-box {{
        background: rgba(5,15,40,0.90);
        border-left: 4px solid #FFD580;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 14px;
    }}
    .step-box h5 {{ color: #FFD580 !important; margin-bottom: 8px; }}
    .formula-box {{
        background: rgba(0,0,0,0.45);
        border: 1px solid rgba(255,210,80,0.5);
        border-radius: 8px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 0.92rem;
        margin: 8px 0;
        color: #FFD580 !important;
    }}
    .highlight-box {{
        background: rgba(255,213,80,0.12);
        border: 2px solid rgba(255,213,80,0.6);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
    }}
    .custom-var-box {{
        background: rgba(0,160,120,0.12);
        border: 1px solid rgba(0,200,150,0.45);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

try:
    set_background("dashb-utama.jpg")
except:
    pass

# LOAD DATASET
@st.cache_data
def load_data():
    df = pd.read_csv("rural_heritage_tourism_industry_chain_dataset.csv")
    hc = df[df["Heritage_Type"] == "Handicraft Center"].copy().reset_index(drop=True)
    hc.index = hc.index + 1
    return df, hc

df_all, df_hc = load_data()

# SESSION STATE
for k, v in [('custom_vars', []), ('custom_rules', []), ('df_hasil', None), ('last_result', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# KONSTANTA VARIABEL & RULE STANDAR
VARS_STANDAR = {
    'visitor': {'label': 'C1 - Visitor Count', 'unit': 'orang', 'tipe': 'Benefit',
        'mf': {'rendah': ('trapmf',[52,52,200,400]), 'sedang': ('trimf',[200,425,650]),
               'tinggi': ('trapmf',[500,650,799,799])}},
    'price': {'label': 'C2 - Ticket Price', 'unit': 'Rp', 'tipe': 'Cost',
        'mf': {'murah': ('trapmf',[10,10,30,55]), 'sedang': ('trimf',[30,55,80]),
               'mahal': ('trapmf',[60,80,99,99])}},
    'satisfaction': {'label': 'C3 - Tourist Satisfaction', 'unit': 'skala', 'tipe': 'Benefit',
        'mf': {'rendah': ('trapmf',[0,0,2,3.25]), 'sedang': ('trimf',[2.5,3.5,4.5]),
               'tinggi': ('trapmf',[3.75,4.5,5,5])}},
    'revenue': {'label': 'C4 - Revenue Generated', 'unit': 'Rp', 'tipe': 'Benefit',
        'mf': {'rendah': ('trapmf',[5000,5000,30000,55000]), 'sedang': ('trimf',[30000,55000,80000]),
               'tinggi': ('trapmf',[60000,80000,100000,100000])}},
    'op_cost': {'label': 'C5 - Operational Cost', 'unit': 'Rp', 'tipe': 'Cost',
        'mf': {'rendah': ('trapmf',[2000,2000,15000,27000]), 'sedang': ('trimf',[15000,27500,40000]),
               'tinggi': ('trapmf',[30000,40000,50000,50000])}},
}

RULES_STANDAR = [
    {"no":"R1", "ant":[("visitor","tinggi"),("satisfaction","tinggi")],
     "cons":"tinggi", "alasan":"Pengunjung banyak + puas → kinerja optimal"},
    {"no":"R2", "ant":[("visitor","sedang"),("satisfaction","tinggi")],
     "cons":"tinggi", "alasan":"Kepuasan tinggi mendominasi walau pengunjung sedang"},
    {"no":"R3", "ant":[("visitor","rendah")],
     "cons":"rendah", "alasan":"Sedikit pengunjung → destinasi tidak diminati"},
    {"no":"R4", "ant":[("price","murah"),("satisfaction","tinggi")],
     "cons":"tinggi", "alasan":"Harga terjangkau + kepuasan tinggi → value for money"},
    {"no":"R5", "ant":[("price","mahal"),("satisfaction","rendah")],
     "cons":"rendah", "alasan":"Harga mahal tidak sebanding pengalaman"},
    {"no":"R6", "ant":[("price","sedang"),("satisfaction","sedang")],
     "cons":"sedang", "alasan":"Kondisi rata-rata → kinerja cukup"},
    {"no":"R7", "ant":[("revenue","tinggi"),("op_cost","rendah")],
     "cons":"tinggi", "alasan":"Pendapatan besar + biaya rendah → profit tinggi"},
    {"no":"R8", "ant":[("revenue","tinggi"),("op_cost","tinggi")],
     "cons":"sedang", "alasan":"Pendapatan besar tapi biaya tinggi"},
    {"no":"R9", "ant":[("revenue","rendah")],
     "cons":"rendah", "alasan":"Pendapatan kecil → tidak menguntungkan"},
    {"no":"R10", "ant":[("op_cost","tinggi")],
     "cons":"rendah", "alasan":"Biaya tinggi menekan kinerja"},
    {"no":"R11", "ant":[("op_cost","rendah"),("satisfaction","tinggi")],
     "cons":"tinggi", "alasan":"Efisiensi biaya + kepuasan tinggi → unggul"},
    {"no":"R12", "ant":[("satisfaction","rendah")],
     "cons":"rendah", "alasan":"Kepuasan rendah → layanan buruk"},
    {"no":"R13", "ant":[("satisfaction","sedang")],
     "cons":"sedang", "alasan":"Kepuasan biasa → kinerja proporsional"},
    {"no":"R14", "ant":[("visitor","tinggi"),("price","mahal")],
     "cons":"sedang", "alasan":"Banyak pengunjung meski mahal → mixed signal"},
    {"no":"R15", "ant":[("visitor","sedang"),("price","sedang")],
     "cons":"sedang", "alasan":"Kondisi moderat di semua aspek"},
]

# FUZZY SYSTEM CACHE
_fs_cache = {}

def _cache_key(extra_vars, extra_rules):
    import hashlib, json
    return hashlib.md5(
        json.dumps({"v": extra_vars or [], "r": extra_rules or []}, sort_keys=True, default=str).encode()
    ).hexdigest()

def get_fuzzy_system(extra_vars=None, extra_rules=None):
    key = _cache_key(extra_vars, extra_rules)
    if key not in _fs_cache:
        _fs_cache[key] = _build_fuzzy_system(extra_vars, extra_rules)
    return _fs_cache[key]

def _build_fuzzy_system(extra_vars=None, extra_rules=None):
    kinerja = ctrl.Consequent(np.arange(0, 101, 1), 'kinerja', defuzzify_method='centroid')
    kinerja['rendah'] = fuzz.trapmf(kinerja.universe, [0, 0, 25, 50])
    kinerja['sedang'] = fuzz.trimf(kinerja.universe, [25, 50, 75])
    kinerja['tinggi'] = fuzz.trapmf(kinerja.universe, [50, 75, 100, 100])

    vis = ctrl.Antecedent(np.arange(52, 800, 1), 'visitor')
    prc = ctrl.Antecedent(np.arange(10, 100, 1), 'price')
    sat = ctrl.Antecedent(np.arange(0.0, 5.1, 0.1), 'satisfaction')
    rev = ctrl.Antecedent(np.arange(5000, 100001, 100), 'revenue')
    opc = ctrl.Antecedent(np.arange(2000, 50001, 100), 'op_cost')

    vis['rendah'] = fuzz.trapmf(vis.universe, [52,52,200,400]); vis['sedang'] = fuzz.trimf(vis.universe, [200,425,650]); vis['tinggi'] = fuzz.trapmf(vis.universe, [500,650,799,799])
    prc['murah'] = fuzz.trapmf(prc.universe, [10,10,30,55]); prc['sedang'] = fuzz.trimf(prc.universe, [30,55,80]); prc['mahal'] = fuzz.trapmf(prc.universe, [60,80,99,99])
    sat['rendah'] = fuzz.trapmf(sat.universe, [0,0,2,3.25]); sat['sedang'] = fuzz.trimf(sat.universe, [2.5,3.5,4.5]); sat['tinggi'] = fuzz.trapmf(sat.universe, [3.75,4.5,5,5])
    rev['rendah'] = fuzz.trapmf(rev.universe, [5000,5000,30000,55000]); rev['sedang'] = fuzz.trimf(rev.universe, [30000,55000,80000]); rev['tinggi'] = fuzz.trapmf(rev.universe, [60000,80000,100000,100000])
    opc['rendah'] = fuzz.trapmf(opc.universe, [2000,2000,15000,27000]); opc['sedang'] = fuzz.trimf(opc.universe, [15000,27500,40000]); opc['tinggi'] = fuzz.trapmf(opc.universe, [30000,40000,50000,50000])

    all_vars = {'visitor': vis, 'price': prc, 'satisfaction': sat, 'revenue': rev, 'op_cost': opc}

    if extra_vars:
        for cv in extra_vars:
            u = np.arange(cv['lo'], cv['hi'] + cv['step'], cv['step'])
            a = ctrl.Antecedent(u, cv['key'])
            _assign_mf(a, cv['term_rendah'], cv['mf_type_rendah'], cv['mf_rendah'])
            _assign_mf(a, cv['term_sedang'], cv['mf_type_sedang'], cv['mf_sedang'])
            _assign_mf(a, cv['term_tinggi'], cv['mf_type_tinggi'], cv['mf_tinggi'])
            all_vars[cv['key']] = a

    rules = [
        ctrl.Rule(vis['tinggi'] & sat['tinggi'], kinerja['tinggi']),
        ctrl.Rule(vis['sedang'] & sat['tinggi'], kinerja['tinggi']),
        ctrl.Rule(vis['rendah'], kinerja['rendah']),
        ctrl.Rule(prc['murah'] & sat['tinggi'], kinerja['tinggi']),
        ctrl.Rule(prc['mahal'] & sat['rendah'], kinerja['rendah']),
        ctrl.Rule(prc['sedang'] & sat['sedang'], kinerja['sedang']),
        ctrl.Rule(rev['tinggi'] & opc['rendah'], kinerja['tinggi']),
        ctrl.Rule(rev['tinggi'] & opc['tinggi'], kinerja['sedang']),
        ctrl.Rule(rev['rendah'], kinerja['rendah']),
        ctrl.Rule(opc['tinggi'], kinerja['rendah']),
        ctrl.Rule(opc['rendah'] & sat['tinggi'], kinerja['tinggi']),
        ctrl.Rule(sat['rendah'], kinerja['rendah']),
        ctrl.Rule(sat['sedang'], kinerja['sedang']),
        ctrl.Rule(vis['tinggi'] & prc['mahal'], kinerja['sedang']),
        ctrl.Rule(vis['sedang'] & prc['sedang'], kinerja['sedang']),
    ]

    if extra_rules:
        for cr in extra_rules:
            try:
                if cr['var_key'] in all_vars:
                    rules.append(ctrl.Rule(all_vars[cr['var_key']][cr['var_term']], kinerja[cr['cons']]))
            except Exception:
                pass

    return ctrl.ControlSystem(rules), kinerja, all_vars

def _assign_mf(ant, term, mf_type, params):
    if mf_type == 'trapmf':
        ant[term] = fuzz.trapmf(ant.universe, params)
    else:
        ant[term] = fuzz.trimf(ant.universe, params)

# HITUNG KINERJA — SATU FUNGSI, SEMUA KEBUTUHAN
def hitung_kinerja(inputs_dict):
    try:
        ev = st.session_state.get('custom_vars', [])
        er = st.session_state.get('custom_rules', [])
        ks, kinerja_var, all_vars = get_fuzzy_system(ev, er)
        sim = ctrl.ControlSystemSimulation(ks)
        for k, v in inputs_dict.items():
            if k in all_vars:
                sim.input[k] = float(v)
        sim.compute()
        nilai = sim.output['kinerja']
        label = "🟢 Tinggi" if nilai >= 66.67 else ("🟡 Sedang" if nilai >= 33.33 else "🔴 Rendah")
        mbd = _get_membership(inputs_dict, all_vars)
        all_rules_def = RULES_STANDAR + er
        firing = _get_firing(mbd, all_rules_def)
        u_out, agg, agg_r, agg_s, agg_t = _aggregate(firing, kinerja_var)
        return {
            'nilai': nilai, 'label': label,
            'kinerja_var': kinerja_var, 'all_vars': all_vars,
            'mbd': mbd, 'firing': firing,
            'u_out': u_out, 'agg': agg,
            'agg_r': agg_r, 'agg_s': agg_s, 'agg_t': agg_t,
            'inputs': dict(inputs_dict),
        }
    except Exception as e:
        return {'nilai': None, 'label': f"Error: {e}"}

def _get_membership(inputs_dict, all_vars):
    res = {}
    for k, var_obj in all_vars.items():
        if k not in inputs_dict:
            continue
        val = float(inputs_dict[k])
        res[k] = {term: round(float(fuzz.interp_membership(var_obj.universe, var_obj[term].mf, val)), 6)
                   for term in var_obj.terms}
    return res

def _get_firing(mbd, all_rules_def):
    res = []
    for r in all_rules_def:
        alphas, ant_strs, valid = [], [], True
        for (vk, term) in r['ant']:
            if vk in mbd and term in mbd[vk]:
                alphas.append(mbd[vk][term])
                lbl = VARS_STANDAR[vk]['label'] if vk in VARS_STANDAR else vk
                ant_strs.append(f"{lbl} [{term.upper()}]")
            else:
                valid = False; break
        alpha = min(alphas) if valid and alphas else 0.0
        comp = f"min({', '.join(f'{a:.4f}' for a in alphas)})" if len(alphas) > 1 else (f"{alphas[0]:.4f}" if alphas else "0")
        res.append({'no': r['no'], 'antecedent': " ∧ ".join(ant_strs) or "—",
                     'comp': comp, 'alpha': round(alpha, 6),
                     'konsekuen': r['cons'].capitalize(),
                     'alasan': r.get('alasan', '')})
    return res

def _aggregate(firing, kinerja_var):
    u = kinerja_var.universe
    ar = np.zeros_like(u, dtype=float)
    as_ = np.zeros_like(u, dtype=float)
    at = np.zeros_like(u, dtype=float)
    for r in firing:
        k = r['konsekuen'].lower()
        clp = np.fmin(r['alpha'], kinerja_var[k].mf)
        if k == 'rendah': ar = np.fmax(ar, clp)
        elif k == 'sedang': as_ = np.fmax(as_, clp)
        else: at = np.fmax(at, clp)
    return u, np.fmax(ar, np.fmax(as_, at)), ar, as_, at

def _centroid_detail(u_out, agg):
    num = float(np.sum(u_out * agg))
    den = float(np.sum(agg))
    cog = num / den if den != 0 else 0.0
    idx = np.linspace(0, len(u_out)-1, 12, dtype=int)
    tbl = pd.DataFrame({
        'y (universe)': u_out[idx],
        'μ_agg(y)': [round(float(agg[i]),4) for i in idx],
        'y × μ_agg(y)': [round(float(u_out[i]*agg[i]),4) for i in idx]
    })
    return cog, num, den, tbl

def _mom(u, agg):
    mv = np.max(agg); return float(np.mean(u[agg >= mv-1e-9])) if mv else 0.0

def _lom(u, agg):
    mv = np.max(agg); idx = np.where(agg >= mv-1e-9)[0]; return float(u[idx[-1]]) if mv else 0.0

def _som(u, agg):
    mv = np.max(agg); idx = np.where(agg >= mv-1e-9)[0]; return float(u[idx[0]]) if mv else 0.0

# HELPER: MF MANUAL (untuk perhitungan langkah demi langkah)
def hitung_mf_manual(x, mf_type, params):
    x = float(x)
    if mf_type == 'trapmf':
        a,b,c,d = params
        if x <= a: return 1.0 if a == b else 0.0
        if x >= d: return 1.0 if c == d else 0.0
        if a < x <= b: return (x-a)/(b-a) if b != a else 1.0
        if b <= x <= c: return 1.0
        if c < x < d: return (d-x)/(d-c)
    else:
        a,b,c = params
        if x <= a or x >= c: return 0.0
        if a < x <= b: return (x-a)/(b-a) if b != a else 1.0
        if b < x < c: return (c-x)/(c-b)
    return 0.0

def html_substitusi_mf(var_label, nilai, mf_dict):
    """Buat HTML perhitungan manual substitusi nilai ke rumus MF."""
    html = f"<b>{var_label}</b> &nbsp;→&nbsp; x = <b style='color:#FFD580;'>{nilai:,}</b><br><br>"
    for term, (tp, prm) in mf_dict.items():
        mu = hitung_mf_manual(nilai, tp, prm)
        html += f"<u>Himpunan <b>{term.capitalize()}</b></u> ({tp}):<br>"
        x = float(nilai)
        if tp == 'trapmf':
            a,b,c,d = prm
            if x <= a:
                html += f"&nbsp;&nbsp;x={x} ≤ a={a} → μ = <b>{'1.0' if a==b else '0.0'}</b><br>"
            elif x >= d:
                html += f"&nbsp;&nbsp;x={x} ≥ d={d} → μ = <b>{'1.0' if c==d else '0.0'}</b><br>"
            elif a < x <= b:
                numer = x-a; denom = b-a
                html += (f"&nbsp;&nbsp;a={a} &lt; x={x} ≤ b={b} → "
                         f"μ = (x−a)/(b−a) = ({x}−{a})/({b}−{a}) = {numer:.4f}/{denom:.4f} = <b style='color:#FFD580;'>{mu:.4f}</b><br>")
            elif b <= x <= c:
                html += f"&nbsp;&nbsp;b={b} ≤ x={x} ≤ c={c} → zona plateau → μ = <b style='color:#FFD580;'>1.0</b><br>"
            elif c < x < d:
                numer = d-x; denom = d-c
                html += (f"&nbsp;&nbsp;c={c} &lt; x={x} &lt; d={d} → "
                         f"μ = (d−x)/(d−c) = ({d}−{x})/({d}−{c}) = {numer:.4f}/{denom:.4f} = <b style='color:#FFD580;'>{mu:.4f}</b><br>")
        else:
            a,b,c = prm
            if x <= a or x >= c:
                html += f"&nbsp;&nbsp;x={x} ≤ a={a} atau ≥ c={c} → di luar rentang → μ = <b style='color:#FFD580;'>0.0</b><br>"
            elif a < x <= b:
                numer = x-a; denom = b-a
                html += (f"&nbsp;&nbsp;a={a} &lt; x={x} ≤ b={b} → "
                         f"μ = (x−a)/(b−a) = ({x}−{a})/({b}−{a}) = {numer:.4f}/{denom:.4f} = <b style='color:#FFD580;'>{mu:.4f}</b><br>")
            elif b < x < c:
                numer = c-x; denom = c-b
                html += (f"&nbsp;&nbsp;b={b} &lt; x={x} &lt; c={c} → "
                         f"μ = (c−x)/(c−b) = ({c}−{x})/({c}−{b}) = {numer:.4f}/{denom:.4f} = <b style='color:#FFD580;'>{mu:.4f}</b><br>")
        html += "<br>"
    return html

# PLOT HELPERS
DARK_BG = "#0a1932"
CLR = ["#ff4b4b", "#ffb000", "#00d084"]
GOLD = "#FFD580"

def _style(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor(GOLD)

def fig_mf(var_obj, title):
    fig, ax = plt.subplots(figsize=(6,3)); fig.patch.set_facecolor(DARK_BG)
    for i,t in enumerate(var_obj.terms):
        ax.plot(var_obj.universe, var_obj[t].mf, lw=2.5, label=t.capitalize(), color=CLR[i%3])
    _style(ax, title)
    leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=8)
    for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout(); return fig

def fig_fuzz_input(var_obj, value, title, lbl):
    fig, ax = plt.subplots(figsize=(6,3)); fig.patch.set_facecolor(DARK_BG)
    for i,term in enumerate(var_obj.terms):
        ax.plot(var_obj.universe, var_obj[term].mf, lw=2, label=term.capitalize(), color=CLR[i%3])
        mu = float(fuzz.interp_membership(var_obj.universe, var_obj[term].mf, float(value)))
        if mu > 0:
            ax.plot([value,value],[0,mu],'--', color=CLR[i%3], lw=1.2, alpha=0.8)
            ax.plot(value, mu, 'o', color=CLR[i%3], ms=5)
            ax.annotate(f'μ={mu:.4f}', xy=(value,mu), xytext=(8,4), textcoords='offset points', fontsize=7, color=CLR[i%3])
    ax.axvline(x=value, color='white', ls=':', lw=1.5, alpha=0.6, label=f"{lbl}={value}")
    _style(ax, title)
    leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=7)
    for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout(); return fig

def fig_implication(u, ar, as_, at, agg, nilai, kv):
    fig, axes = plt.subplots(1,4,figsize=(16,3.5)); fig.patch.set_facecolor(DARK_BG)
    for ax,(title,arr,fc) in zip(axes,[("Implication: Rendah",ar,"#ff4b4b"),
                                        ("Implication: Sedang",as_,"#ffb000"),
                                        ("Implication: Tinggi",at,"#00d084"),
                                        ("Agregasi + Defuzz",agg,"#5599ff")]):
        ax.fill_between(u, arr, alpha=0.4, color=fc); ax.plot(u, arr, color=fc, lw=2)
        for i,t in enumerate(kv.terms): ax.plot(u, kv[t].mf, '--', color=CLR[i%3], lw=0.8, alpha=0.4)
        _style(ax, title)
        if "Defuzz" in title:
            ax.axvline(x=nilai, color='yellow', lw=2, ls='--', label=f"Centroid={nilai:.2f}")
            leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=7)
            for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout(); return fig

def fig_clipping(aktif, u, kv):
    n = min(len(aktif), 6)
    if n == 0: return None
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3.2)); fig.patch.set_facecolor(DARK_BG)
    if n == 1: axes = [axes]
    cm = {'rendah':'#ff4b4b','sedang':'#ffb000','tinggi':'#00d084'}
    for ax, r in zip(axes, aktif[:6]):
        k = r['konsekuen'].lower(); alp = r['alpha']
        clp = np.fmin(alp, kv[k].mf)
        ax.plot(u, kv[k].mf, '--', color=cm[k], lw=1.5, alpha=0.5, label='MF asli')
        ax.fill_between(u, clp, alpha=0.5, color=cm[k])
        ax.plot(u, clp, color=cm[k], lw=2, label=f'α={alp:.4f}')
        ax.axhline(y=alp, color='yellow', ls=':', lw=1, alpha=0.8)
        _style(ax, f"{r['no']}: {r['konsekuen']}")
        leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=7)
        for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout(); return fig

def fig_defuzz_cmp(u, agg, cog, mom, lom, som):
    fig, ax = plt.subplots(figsize=(10,4)); fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.fill_between(u, agg, alpha=0.25, color='#5599ff', label='Area Agregasi')
    ax.plot(u, agg, color='#5599ff', lw=2)
    for name, val, clr in [("Centroid (CoG)",cog,'yellow'),("MOM",mom,'#00ff88'),("LOM",lom,'#ff8800'),("SOM",som,'#ff4488')]:
        ax.axvline(x=val, color=clr, ls='--', lw=1.8, label=f"{name} = {val:.4f}")
    _style(ax, "Perbandingan Metode Defuzzifikasi")
    ax.set_xlabel("Output (0–100)", color='white'); ax.set_ylabel("μ_agg(y)", color='white')
    leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=8)
    for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout(); return fig

@st.cache_data(show_spinner=False)
def fig_sensitivity_cached(base_tuple, param_key, range_tuple, label_x, cfg_key):
    base = dict(base_tuple); rng = list(range_tuple)
    ev = st.session_state.get('custom_vars', []); er = st.session_state.get('custom_rules', [])
    ks, _, all_vars = get_fuzzy_system(ev, er)
    sim = ctrl.ControlSystemSimulation(ks)
    vals, outs = [], []
    for v in rng:
        inp = {**base, param_key: float(v)}
        try:
            for k, val in inp.items():
                if k in all_vars: sim.input[k] = float(val)
            sim.compute(); vals.append(v); outs.append(sim.output['kinerja'])
        except: pass
    fig, ax = plt.subplots(figsize=(8,4)); fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.plot(vals, outs, color='#00d0ff', lw=2.5)
    ax.fill_between(vals, outs, alpha=0.15, color='#00d0ff')
    ax.axhline(y=66.67, color='#2ecc71', ls='--', lw=1, alpha=0.7, label='Batas Tinggi (66.67)')
    ax.axhline(y=33.33, color='#e74c3c', ls='--', lw=1, alpha=0.7, label='Batas Rendah (33.33)')
    _style(ax, f"Sensitivitas: {label_x} vs Kinerja")
    ax.set_xlabel(label_x, color='white'); ax.set_ylabel("Nilai Kinerja", color='white')
    leg = ax.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=8)
    for t in leg.get_texts(): t.set_color("white")
    plt.tight_layout()
    return fig, vals, outs

# UI HELPERS
def infobox(html):
    st.markdown(f"<div class='info-box'>{html}</div>", unsafe_allow_html=True)

def stepbox(title, html):
    st.markdown(f"<div class='step-box'><h5>{title}</h5>{html}</div>", unsafe_allow_html=True)

def highlightbox(html):
    st.markdown(f"<div class='highlight-box'>{html}</div>", unsafe_allow_html=True)

def render_fig(f):
    st.pyplot(f); plt.close(f)

def label_color(cons):
    return '#2ecc71' if cons=='tinggi' else '#f39c12' if cons=='sedang' else '#e74c3c'

# HEADER
st.markdown("""
<div style='text-align:center; padding:22px 0 10px 0;'>
    <h1 style='font-size:2.2rem; margin-bottom:6px; color:#FFFFFF;'>
        SPK Penilaian Kinerja Destinasi Wisata Kerajinan
    </h1>
    <div style='display:inline-block; background:rgba(10,25,55,0.85);
                border:1px solid rgba(255,210,80,0.45);
                border-radius:20px; padding:5px 22px;'>
        <span style='color:#FFD580; font-size:1.05rem; font-weight:600;'>
            Handicraft Tourism &nbsp;|&nbsp; Fuzzy Logic Mamdani
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dataset",
    "⚙️ Hitung SPK",
    "📊 Fungsi Keanggotaan",
    "🛠 Kelola Kriteria",
    "📈 Analisis Sensitivitas",
    "👥 Profil Kelompok",
])

# TAB 1 — DATASET
with tab1:
    infobox(f"""
    Dataset yang digunakan adalah <b>Rural Heritage Tourism Industry Chain Dataset</b>
    yang difilter pada <b>Heritage_Type = Handicraft Center</b>.<br>
    Dataset memiliki <b>784 baris</b> data dengan <b>5 kriteria</b> utama yang digunakan
    dalam perhitungan SPK.
    """)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Visitor Count", f"{df_hc['Visitor_Count'].mean():.0f}", "rata-rata orang")
    c2.metric("Ticket Price", f"Rp {df_hc['Ticket_Price'].mean():.0f}", "rata-rata")
    c3.metric("Tourist Satisf.", f"{df_hc['Tourist_Satisfaction'].mean():.2f}", "rata-rata (1–5)")
    c4.metric("Revenue Generated", f"Rp {df_hc['Revenue_Generated'].mean():,.0f}","rata-rata")
    c5.metric("Operational Cost", f"Rp {df_hc['Operational_Cost'].mean():,.0f}", "rata-rata")

    with st.expander("Statistik Deskriptif Lengkap"):
        cols_s = ['Visitor_Count','Ticket_Price','Tourist_Satisfaction','Revenue_Generated','Operational_Cost']
        df_stat = df_hc[cols_s].describe().T.round(4)
        df_stat.columns = ['N','Mean','Std','Min','Q1','Median','Q3','Max']
        st.dataframe(df_stat, use_container_width=True)

    infobox(f"<b>Tabel Dataset</b> — {len(df_hc)} baris")
    st.dataframe(df_hc[['Location_ID','Visitor_Count','Ticket_Price',
                         'Tourist_Satisfaction','Revenue_Generated','Operational_Cost']],
                 use_container_width=True, height=380)

    with st.expander("Keterangan Kolom & Range Data"):
        infobox("""
        <b>Kriteria dalam SPK:</b><br><br>
        <table style='width:100%;border-collapse:collapse;color:#FFF;'>
        <tr style='border-bottom:1px solid rgba(255,210,80,.4);'>
            <th style='padding:6px 10px;color:#FFD580;'>Kode</th>
            <th style='padding:6px 10px;color:#FFD580;'>Kolom</th>
            <th style='padding:6px 10px;color:#FFD580;'>Keterangan</th>
            <th style='padding:6px 10px;color:#FFD580;'>Tipe</th>
            <th style='padding:6px 10px;color:#FFD580;'>Range</th>
        </tr>
        <tr><td style='padding:5px 10px;'>C1</td><td>Visitor_Count</td><td>Jumlah pengunjung</td><td>Benefit ↑</td><td>52–799</td></tr>
        <tr><td style='padding:5px 10px;'>C2</td><td>Ticket_Price</td><td>Harga tiket (Rp)</td><td>Cost ↓</td><td>10–99</td></tr>
        <tr><td style='padding:5px 10px;'>C3</td><td>Tourist_Satisfaction</td><td>Kepuasan wisatawan</td><td>Benefit ↑</td><td>2.5–5.0</td></tr>
        <tr><td style='padding:5px 10px;'>C4</td><td>Revenue_Generated</td><td>Pendapatan (Rp)</td><td>Benefit ↑</td><td>5.125–99.979</td></tr>
        <tr><td style='padding:5px 10px;'>C5</td><td>Operational_Cost</td><td>Biaya operasional (Rp)</td><td>Cost ↓</td><td>2.081–49.726</td></tr>
        <tr style='border-top:1px solid rgba(255,210,80,.4);'>
            <td style='padding:5px 10px;'>Output</td><td>Kinerja</td><td>Hasil fuzzy Mamdani</td><td>—</td><td>0–100</td></tr>
        </table>
        """)

# TAB 2 — HITUNG SPK (input + hasil + detail Mamdani per-expander)
with tab2:
    ev = st.session_state.get('custom_vars', [])
    er = st.session_state.get('custom_rules', [])

    infobox(f"""
    <h4>Hitung SPK — Fuzzy Logic Mamdani</h4>
    Masukkan nilai kriteria, klik <b>Hitung</b>, lalu hasil dan <b>seluruh proses Mamdani 5 langkah</b>
    tampil di bawah ini. Klik setiap expander untuk membuka detail perhitungan.<br>
    Mode <b>Batch</b> tersedia untuk menghitung semua {len(df_hc)} baris dataset sekaligus.
    """)

    mode = st.radio("Mode:", ["Input Manual (1 Destinasi)", "Batch Dataset (Semua Data)"], horizontal=True)

    # INPUT MANUAL
    if mode == "Input Manual (1 Destinasi)":
        st.markdown("#### Input Nilai Kriteria")
        ca, cb = st.columns(2)
        with ca:
            v_vis = st.slider("C1 – Visitor Count (orang)", 52, 799, 418, 1)
            v_prc = st.slider("C2 – Ticket Price (Rp)", 10, 99, 53, 1)
            v_sat = st.slider("C3 – Tourist Satisfaction", 0.0, 5.0, 3.7, 0.1)
        with cb:
            v_rev = st.number_input("C4 – Revenue Generated (Rp)", 5000, 100000, 52000, 1000)
            v_opc = st.number_input("C5 – Operational Cost (Rp)", 2000, 50000, 25000, 500)

        inputs = {'visitor': v_vis, 'price': v_prc, 'satisfaction': v_sat,
                  'revenue': v_rev, 'op_cost': v_opc}

        if ev:
            st.markdown(f"**Kriteria Tambahan ({len(ev)}):**")
            exc = st.columns(min(len(ev), 3))
            for i, cv in enumerate(ev):
                with exc[i % 3]:
                    inputs[cv['key']] = st.number_input(
                        f"{cv['label']} ({cv['tipe']})",
                        float(cv['lo']), float(cv['hi']),
                        round((cv['lo']+cv['hi'])/2, 4),
                        float(cv['step']),
                        key=f"t2_cv_{cv['key']}")

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("▶ Hitung Kinerja Destinasi", type="primary"):
            with st.spinner("Menghitung..."):
                res = hitung_kinerja(inputs)
                st.session_state['last_result'] = res

        res = st.session_state.get('last_result')
        if res and res.get('nilai') is not None:
            nilai = res['nilai']; label = res['label']
            kv = res['kinerja_var']; all_vars = res['all_vars']
            mbd = res['mbd']; firing = res['firing']
            u_out = res['u_out']; agg = res['agg']
            agg_r = res['agg_r']; agg_s = res['agg_s']; agg_t = res['agg_t']
            saved_inputs = res['inputs']
            aktif = [r for r in firing if r['alpha'] > 0]

            # HASIL RINGKAS
            highlightbox(f"""
            <h4 style='margin:0;color:#FFD580;'>✅ Hasil Perhitungan</h4>
            <p style='font-size:1.5rem;margin:8px 0;'>
                Nilai Kinerja: <b style='color:#FFD580;font-size:1.9rem;'>{nilai:.4f}</b>
                &nbsp;|&nbsp; Kategori: <b style='font-size:1.3rem;'>{label}</b>
            </p>
            <p style='color:rgba(255,255,255,0.7);font-size:0.85rem;'>
                Metode: Centroid (CoG) &nbsp;|&nbsp; Operator: AND=min, OR=max &nbsp;|&nbsp;
                Rule: {len(RULES_STANDAR)+len(er)} rule &nbsp;|&nbsp; Rentang Output: 0–100
            </p>
            """)

            mc = st.columns(7)
            mc[0].metric("C1 Visitor", v_vis)
            mc[1].metric("C2 Price", f"Rp {v_prc}")
            mc[2].metric("C3 Satisf.", v_sat)
            mc[3].metric("C4 Revenue", f"Rp {v_rev:,}")
            mc[4].metric("C5 Op.Cost", f"Rp {v_opc:,}")
            mc[5].metric("Kinerja", f"{nilai:.4f}")
            mc[6].metric("Kategori", label.split()[-1])

            # Grafik output ringkas
            infobox("<b>Kurva Output – Area Agregasi & Posisi Centroid</b>")
            fig_out, ax_o = plt.subplots(figsize=(14,4)); fig_out.patch.set_facecolor(DARK_BG); ax_o.set_facecolor(DARK_BG)
            ax_o.fill_between(u_out, agg, alpha=0.35, color="#5599ff", label="Area Agregasi")
            ax_o.plot(u_out, agg, color="#5599ff", lw=2)
            for i,t in enumerate(kv.terms):
                ax_o.plot(u_out, kv[t].mf, '--', color=CLR[i%3], lw=1.2, alpha=0.6, label=f"MF {t.capitalize()}")
            ax_o.axvline(x=nilai, color='yellow', ls='--', lw=2.5, label=f"Centroid={nilai:.4f}")
            ax_o.axvspan(0,33.33,alpha=0.06,color='#ff4b4b',label='Zona Rendah')
            ax_o.axvspan(33.33,66.67,alpha=0.06,color='#ffb000',label='Zona Sedang')
            ax_o.axvspan(66.67,100,alpha=0.06,color='#00d084',label='Zona Tinggi')
            _style(ax_o, "Output Kinerja Destinasi — Area Agregasi & Defuzzifikasi Centroid")
            leg = ax_o.legend(facecolor="#0d1f40", edgecolor=GOLD, fontsize=8, loc='upper left')
            for t in leg.get_texts(): t.set_color("white")
            plt.tight_layout(); render_fig(fig_out)

            # DETAIL MAMDANI — 5 expander terpisah
            st.markdown("---")
            st.markdown("### 🔍 Detail Proses Mamdani Step-by-Step")
            st.caption("Klik setiap langkah untuk melihat detail perhitungan dan visualisasi.")

            # STEP 1: FUZZIFIKASI
            with st.expander("① FUZZIFIKASI — Konversi Nilai Crisp → Derajat Keanggotaan μ", expanded=False):
                stepbox("Konsep & Rumus",
                """
                Fuzzifikasi adalah proses mengubah nilai input crisp (angka pasti) menjadi
                <b>derajat keanggotaan (μ)</b> pada setiap himpunan fuzzy menggunakan fungsi keanggotaan.<br><br>
                <b>Rumus trapmf [a,b,c,d]:</b><br>
                <div class='formula-box'>
                μ(x) = 0,&nbsp; jika x ≤ a atau x ≥ d<br>
                μ(x) = (x−a)/(b−a),&nbsp; jika a &lt; x ≤ b<br>
                μ(x) = 1,&nbsp; jika b ≤ x ≤ c &nbsp;(zona plateau)<br>
                μ(x) = (d−x)/(d−c),&nbsp; jika c &lt; x &lt; d
                </div>
                <b>Rumus trimf [a,b,c]:</b><br>
                <div class='formula-box'>
                μ(x) = 0,&nbsp; jika x ≤ a atau x ≥ c<br>
                μ(x) = (x−a)/(b−a),&nbsp; jika a &lt; x ≤ b<br>
                μ(x) = (c−x)/(c−b),&nbsp; jika b &lt; x &lt; c
                </div>
                """)

                # Tabel ringkasan semua variabel
                infobox("<b>Tabel Derajat Keanggotaan — Semua Variabel Input:</b>")
                rows_f = []
                for vk, vi in VARS_STANDAR.items():
                    if vk not in mbd: continue
                    d = mbd[vk]; tms = list(d.keys())
                    rows_f.append({"Variabel": vi['label'],
                                    "Nilai Input": f"{saved_inputs.get(vk,'—'):,}",
                                    f"μ {tms[0].capitalize()}": d[tms[0]],
                                    f"μ {tms[1].capitalize()}": d[tms[1]],
                                    f"μ {tms[2].capitalize()}": d[tms[2]]})
                for cv in ev:
                    if cv['key'] not in mbd: continue
                    d = mbd[cv['key']]; tms = list(d.keys())
                    rows_f.append({"Variabel": cv['label'],
                                    "Nilai Input": str(saved_inputs.get(cv['key'],'—')),
                                    f"μ {tms[0].capitalize()}": d[tms[0]],
                                    f"μ {tms[1].capitalize()}": d[tms[1]],
                                    f"μ {tms[2].capitalize()}": d[tms[2]]})
                st.dataframe(pd.DataFrame(rows_f), use_container_width=True)

                # Perhitungan manual substitusi nilai ke rumus
                infobox("<b>Perhitungan Manual — Substitusi Nilai ke Rumus MF:</b>")
                all_disp = [(vk, saved_inputs.get(vk,0), vi['label'],
                             {t: vi['mf'][t] for t in vi['mf']})
                            for vk, vi in VARS_STANDAR.items()]
                for cv in ev:
                    all_disp.append((cv['key'], saved_inputs.get(cv['key'],0), cv['label'], {
                        cv['term_rendah']: (cv['mf_type_rendah'], cv['mf_rendah']),
                        cv['term_sedang']: (cv['mf_type_sedang'], cv['mf_sedang']),
                        cv['term_tinggi']: (cv['mf_type_tinggi'], cv['mf_tinggi']),
                    }))
                mc2 = st.columns(2)
                for idx, (vk, vv, vlbl, mfd) in enumerate(all_disp):
                    with mc2[idx % 2]:
                        html = html_substitusi_mf(vlbl, vv, mfd)
                        st.markdown(f"<div class='info-box' style='font-size:0.85rem;'>{html}</div>", unsafe_allow_html=True)

                # Grafik fuzzifikasi
                st.markdown("**Grafik Fuzzifikasi — Posisi Nilai Input pada Kurva MF:**")
                gcs = st.columns(3)
                disp_vars = [
                    (all_vars['visitor'], v_vis, "C1 – Visitor Count", str(v_vis)),
                    (all_vars['price'], v_prc, "C2 – Ticket Price", str(v_prc)),
                    (all_vars['satisfaction'], v_sat, "C3 – Satisfaction", str(v_sat)),
                    (all_vars['revenue'], v_rev, "C4 – Revenue", str(v_rev)),
                    (all_vars['op_cost'], v_opc, "C5 – Op. Cost", str(v_opc)),
                ]
                for cv in ev:
                    if cv['key'] in all_vars:
                        disp_vars.append((all_vars[cv['key']], saved_inputs.get(cv['key'],0),
                                           cv['label'], str(saved_inputs.get(cv['key'],0))))
                for idx,(vo,val,ttl,lb) in enumerate(disp_vars):
                    with gcs[idx%3]:
                        render_fig(fig_fuzz_input(vo, val, ttl, lb))

            # STEP 2: EVALUASI RULE
            with st.expander("② EVALUASI RULE — Firing Strength α = min(antecedents)", expanded=False):
                stepbox("Konsep & Rumus",
                """
                Setiap rule IF-THEN dievaluasi dengan menghitung <b>firing strength (α)</b>, yaitu
                kekuatan aktivasi rule tersebut berdasarkan nilai input.<br><br>
                <b>Operator AND = minimum:</b><br>
                <div class='formula-box'>α = min(μ_antecedent1, μ_antecedent2, ...)</div>
                Rule dengan α = 0 <b>tidak aktif</b> (tidak berkontribusi pada output).<br>
                Rule dengan α &gt; 0 <b>aktif</b> dan memberikan kontribusi pada konsekuen.
                """)
                rows_r = []
                for r in firing:
                    rows_r.append({"Rule": r['no'],
                                    "IF (Antecedent)": r['antecedent'],
                                    "Perhitungan α": r['comp'],
                                    "α (Firing Strength)": round(r['alpha'],6),
                                    "THEN (Konsekuen)": r['konsekuen'],
                                    "Justifikasi": r['alasan'],
                                    "Status": "✅ Aktif" if r['alpha']>0 else "⬜ Tidak aktif"})
                st.dataframe(pd.DataFrame(rows_r), use_container_width=True)

                infobox(f"""
                <b>Ringkasan Evaluasi Rule:</b><br>
                &nbsp;&nbsp;• Total rule: <b>{len(firing)}</b><br>
                &nbsp;&nbsp;• Rule aktif (α &gt; 0): <b style='color:#2ecc71;'>{len(aktif)}</b><br>
                &nbsp;&nbsp;• Rule tidak aktif (α = 0): <b style='color:#e74c3c;'>{len(firing)-len(aktif)}</b>
                """)

                fig_fs, ax_fs = plt.subplots(figsize=(14,4))
                fig_fs.patch.set_facecolor(DARK_BG); ax_fs.set_facecolor(DARK_BG)
                labs = [r['no'] for r in firing]; alps = [r['alpha'] for r in firing]
                ax_fs.bar(labs, alps, color=['#2ecc71' if a>0 else '#444' for a in alps], edgecolor=GOLD, lw=0.8)
                ax_fs.set_ylim(0, 1.15); ax_fs.set_ylabel("Firing Strength (α)", color='white')
                for i,(l,a) in enumerate(zip(labs,alps)):
                    if a>0: ax_fs.text(i, a+0.02, f"{a:.4f}", ha='center', color='white', fontsize=7)
                _style(ax_fs, "Firing Strength (α) Tiap Rule — Hijau = Aktif, Abu = Tidak Aktif")
                plt.tight_layout(); render_fig(fig_fs)

            # STEP 3: IMPLICATION
            with st.expander("③ IMPLICATION — Clipping MF Konsekuen: μ_clipped(y) = min(α, μ_konsekuen(y))", expanded=False):
                stepbox("Konsep & Rumus",
                """
                Implication (clipping) memotong fungsi keanggotaan konsekuen sesuai firing strength α.<br><br>
                <b>Rumus Clipping:</b><br>
                <div class='formula-box'>μ_clipped(y) = min(α, μ_konsekuen(y))</div>
                Artinya: MF konsekuen "dipotong" di ketinggian α. Area di atas α dihilangkan.<br>
                Setiap rule aktif menghasilkan satu potongan MF yang akan digabungkan di tahap agregasi.
                """)
                infobox("<b>Rule Aktif dan Nilai Clipping-nya:</b><br>" +
                        "".join(f"&nbsp;&nbsp;• <b>{r['no']}</b>: IF {r['antecedent']}"
                                f" → THEN Kinerja <b>{r['konsekuen']}</b>"
                                f" — α = <b style='color:#FFD580;'>{r['alpha']:.6f}</b>"
                                f" → dipotong di ketinggian {r['alpha']:.6f}<br>"
                                for r in aktif) or "<span style='color:#ff8080;'>Tidak ada rule aktif.</span>")
                if aktif:
                    fc = fig_clipping(aktif, u_out, kv)
                    if fc: render_fig(fc)

            # STEP 4: AGREGASI
            with st.expander("④ AGREGASI — Gabungkan Semua Clipped Area: μ_agg(y) = max(semua clipped)", expanded=False):
                stepbox("Konsep & Rumus",
                """
                Agregasi menggabungkan semua area yang sudah di-clip dari seluruh rule aktif
                menjadi <b>satu fungsi agregasi tunggal</b>.<br><br>
                <b>Operator OR = maximum:</b><br>
                <div class='formula-box'>μ_agg(y) = max(μ_clipped_R1(y), μ_clipped_R2(y), ..., μ_clipped_Rn(y))</div>
                Dilakukan per himpunan output (Rendah, Sedang, Tinggi), lalu digabung menjadi satu area.
                """)
                rule_r = [r['no'] for r in firing if r['konsekuen'].lower()=='rendah' and r['alpha']>0]
                rule_s = [r['no'] for r in firing if r['konsekuen'].lower()=='sedang' and r['alpha']>0]
                rule_t = [r['no'] for r in firing if r['konsekuen'].lower()=='tinggi' and r['alpha']>0]
                al_r = [round(r['alpha'],6) for r in firing if r['konsekuen'].lower()=='rendah' and r['alpha']>0]
                al_s = [round(r['alpha'],6) for r in firing if r['konsekuen'].lower()=='sedang' and r['alpha']>0]
                al_t = [round(r['alpha'],6) for r in firing if r['konsekuen'].lower()=='tinggi' and r['alpha']>0]
                st.dataframe(pd.DataFrame({
                    "Himpunan Output": ["Rendah","Sedang","Tinggi"],
                    "Rule Kontributor": [", ".join(rule_r) or "—", ", ".join(rule_s) or "—", ", ".join(rule_t) or "—"],
                    "Nilai α Masing-masing": [str(al_r), str(al_s), str(al_t)],
                    "α Maks (dipakai)": [
                        round(max(al_r, default=0), 6),
                        round(max(al_s, default=0), 6),
                        round(max(al_t, default=0), 6)
                    ],
                    "Keterangan": [
                        "Tinggi dari clipped rendah" if al_r else "Tidak ada rule aktif",
                        "Tinggi dari clipped sedang" if al_s else "Tidak ada rule aktif",
                        "Tinggi dari clipped tinggi" if al_t else "Tidak ada rule aktif",
                    ],
                }), use_container_width=True)
                render_fig(fig_implication(u_out, agg_r, agg_s, agg_t, agg, nilai, kv))

            # STEP 5: DEFUZZIFIKASI
            with st.expander("⑤ DEFUZZIFIKASI — Centroid CoG + Perbandingan MOM/LOM/SOM", expanded=False):
                stepbox("Konsep & Rumus",
                """
                Defuzzifikasi mengubah area agregasi fuzzy kembali menjadi <b>satu nilai crisp output</b>.<br><br>
                <b>Metode Centroid (Center of Gravity / CoG) — yang digunakan:</b><br>
                <div class='formula-box'>y* = Σ[y · μ_agg(y)] / Σ[μ_agg(y)]</div>
                y* adalah titik berat (centroid) dari area agregasi — interpretasi: nilai yang "mewakili"
                keseluruhan area fuzzy secara seimbang.<br><br>
                <b>Metode lain (sebagai perbandingan):</b><br>
                &nbsp;&nbsp;• <b>MOM</b> (Mean of Maximum): rata-rata titik dengan μ tertinggi<br>
                &nbsp;&nbsp;• <b>LOM</b> (Largest of Maximum): titik paling kanan dengan μ tertinggi<br>
                &nbsp;&nbsp;• <b>SOM</b> (Smallest of Maximum): titik paling kiri dengan μ tertinggi
                """)
                cog, num, den, tbl_c = _centroid_detail(u_out, agg)
                mom_v = _mom(u_out, agg); lom_v = _lom(u_out, agg); som_v = _som(u_out, agg)
                infobox(f"""
                <b>Perhitungan Centroid (CoG) Step-by-Step:</b><br>
                &nbsp;&nbsp;• Σ[y · μ_agg(y)] &nbsp;= <b>{num:.6f}</b>
                &nbsp;&nbsp;<small>(jumlah perkalian tiap titik y dengan derajat keanggotaannya)</small><br>
                &nbsp;&nbsp;• Σ[μ_agg(y)] &nbsp;&nbsp;&nbsp;&nbsp;= <b>{den:.6f}</b>
                &nbsp;&nbsp;<small>(total luas area agregasi)</small><br>
                &nbsp;&nbsp;• y* = {num:.6f} / {den:.6f} = <b style='color:#FFD580;font-size:1.1rem;'>{cog:.6f}</b><br>
                &nbsp;&nbsp;• Verifikasi skfuzzy = <b style='color:#FFD580;'>{nilai:.6f}</b>
                &nbsp;&nbsp;✅ selisih = {abs(cog-nilai):.8f}
                """)
                st.markdown("**Sampel Titik Perhitungan Centroid:**")
                st.dataframe(tbl_c, use_container_width=True)

                df_cmp = pd.DataFrame({
                    "Metode": ["Centroid (CoG)","MOM","LOM","SOM"],
                    "Nilai y*": [round(cog,4), round(mom_v,4), round(lom_v,4), round(som_v,4)],
                    "Kategori": ["Tinggi" if v>=66.67 else ("Sedang" if v>=33.33 else "Rendah")
                                 for v in [cog,mom_v,lom_v,som_v]],
                    "Penjelasan": ["Titik berat area — paling umum digunakan",
                                   "Rata-rata titik dengan derajat keanggotaan tertinggi",
                                   "Titik paling kanan dengan derajat keanggotaan tertinggi",
                                   "Titik paling kiri dengan derajat keanggotaan tertinggi"],
                    "Digunakan": ["✅ DIPILIH","","",""],
                })
                infobox("<b>Perbandingan 4 Metode Defuzzifikasi:</b>")
                st.dataframe(df_cmp, use_container_width=True)
                render_fig(fig_defuzz_cmp(u_out, agg, cog, mom_v, lom_v, som_v))

                highlightbox(f"""
                <h4 style='margin:0;color:#FFD580;'>✅ Hasil Akhir Defuzzifikasi</h4>
                <p style='font-size:1.5rem;margin:8px 0;'>
                    y* (Centroid) = <b style='color:#FFD580;font-size:1.9rem;'>{nilai:.4f}</b>
                    &nbsp;→&nbsp; <b style='font-size:1.3rem;'>{label}</b>
                </p>
                <table style='color:#FFF;border-collapse:collapse;'>
                <tr><td style='padding:4px 12px;'>🔴 Rendah: y* &lt; 33.33</td>
                    <td>{'<b style="color:#2ecc71;">← nilai ini termasuk di sini</b>' if nilai<33.33 else ''}</td></tr>
                <tr><td style='padding:4px 12px;'>🟡 Sedang: 33.33 ≤ y* &lt; 66.67</td>
                    <td>{'<b style="color:#2ecc71;">← nilai ini termasuk di sini</b>' if 33.33<=nilai<66.67 else ''}</td></tr>
                <tr><td style='padding:4px 12px;'>🟢 Tinggi: y* ≥ 66.67</td>
                    <td>{'<b style="color:#2ecc71;">← nilai ini termasuk di sini</b>' if nilai>=66.67 else ''}</td></tr>
                </table>
                """)
        elif res and res.get('nilai') is None:
            st.error(f"❌ {res.get('label','Error tidak diketahui')}")

    # BATCH DATASET
    else:
        infobox(f"<b>Dataset:</b> {len(df_hc)} baris siap dihitung sekaligus.")
        if st.button("▶ Jalankan Perhitungan Batch", type="primary"):
            with st.spinner("Membangun sistem fuzzy & menghitung..."):
                ks, _, all_vars_b = get_fuzzy_system(ev, er)
                sim_b = ctrl.ControlSystemSimulation(ks)
                hasil_nilai, hasil_label = [], []
                pb = st.progress(0, text="Menghitung...")
                total = len(df_hc)
                for i, (_, row) in enumerate(df_hc.iterrows()):
                    try:
                        inp_b = {'visitor': row['Visitor_Count'], 'price': row['Ticket_Price'],
                                 'satisfaction': row['Tourist_Satisfaction'],
                                 'revenue': row['Revenue_Generated'], 'op_cost': row['Operational_Cost']}
                        for k, v in inp_b.items():
                            if k in all_vars_b: sim_b.input[k] = float(v)
                        sim_b.compute()
                        n = sim_b.output['kinerja']
                        l = "🟢 Tinggi" if n>=66.67 else ("🟡 Sedang" if n>=33.33 else "🔴 Rendah")
                    except:
                        n, l = None, "Error"
                    hasil_nilai.append(n); hasil_label.append(l)
                    if (i+1) % 50 == 0 or (i+1) == total:
                        pb.progress((i+1)/total, text=f"{i+1}/{total} baris")
                pb.empty()

                df_h = df_hc[['Location_ID','Visitor_Count','Ticket_Price',
                               'Tourist_Satisfaction','Revenue_Generated','Operational_Cost']].copy()
                df_h['Kinerja_Score'] = hasil_nilai; df_h['Kategori'] = hasil_label
                df_h = df_h.dropna(subset=['Kinerja_Score']).sort_values('Kinerja_Score', ascending=False).reset_index(drop=True)
                df_h.insert(0, 'Peringkat', range(1, len(df_h)+1))
                st.session_state['df_hasil'] = df_h
                st.success(f"✅ Selesai! {len(df_h)} destinasi dihitung.")

        if st.session_state.get('df_hasil') is not None:
            df_h = st.session_state['df_hasil']
            jt = sum(1 for k in df_h['Kategori'] if 'Tinggi' in str(k))
            js = sum(1 for k in df_h['Kategori'] if 'Sedang' in str(k))
            jr = sum(1 for k in df_h['Kategori'] if 'Rendah' in str(k))
            infobox(f"🟢 Tinggi: <b>{jt}</b> &nbsp;|&nbsp; 🟡 Sedang: <b>{js}</b> &nbsp;|&nbsp; 🔴 Rendah: <b>{jr}</b>")
            r1,r2,r3 = st.columns(3)
            r1.metric("🟢 Tinggi",jt,"destinasi"); r2.metric("🟡 Sedang",js,"destinasi"); r3.metric("🔴 Rendah",jr,"destinasi")
            st.dataframe(df_h, use_container_width=True, height=420)
            buf = io.StringIO(); df_h.to_csv(buf, index=False)
            st.download_button("⬇ Download CSV", buf.getvalue().encode(), "hasil_spk_handicraft.csv", "text/csv")

            fig_b, ax_b = plt.subplots(figsize=(15,8)); fig_b.patch.set_facecolor(DARK_BG); ax_b.set_facecolor(DARK_BG)
            bars = ax_b.bar(['Rendah','Sedang','Tinggi'],[jr,js,jt],color=['#e74c3c','#f39c12','#2ecc71'],edgecolor=GOLD,lw=1.2)
            for bar, val in zip(bars,[jr,js,jt]):
                ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(val),
                           ha='center', va='bottom', color='white', fontweight='bold', fontsize=13)
            _style(ax_b, "Distribusi Kategori Kinerja Destinasi"); ax_b.set_ylabel("Jumlah Destinasi", color='white')
            plt.tight_layout(); render_fig(fig_b)

# TAB 3 — FUNGSI KEANGGOTAAN
with tab3:
    infobox("""
    <h4>Fungsi Keanggotaan (Membership Function)</h4>
    Setiap variabel dibagi 3 himpunan fuzzy: <b>Trapesium (trapmf)</b> untuk himpunan tepi,
    <b>Segitiga (trimf)</b> untuk himpunan tengah.
    """)
    ev_t3 = st.session_state.get('custom_vars', [])
    try:
        _, kv_t3, av_t3 = get_fuzzy_system(ev_t3, [])
        k1, k2 = st.columns(2)
        with k1:
            render_fig(fig_mf(av_t3['visitor'], VARS_STANDAR['visitor']['label']))
            render_fig(fig_mf(av_t3['satisfaction'], VARS_STANDAR['satisfaction']['label']))
            render_fig(fig_mf(av_t3['op_cost'], VARS_STANDAR['op_cost']['label']))
        with k2:
            render_fig(fig_mf(av_t3['price'], VARS_STANDAR['price']['label']))
            render_fig(fig_mf(av_t3['revenue'], VARS_STANDAR['revenue']['label']))
            render_fig(fig_mf(kv_t3, "Output – Kinerja Destinasi"))
    except Exception as e:
        st.error(f"Error: {e}")

    infobox("<b>Parameter MF & Rumus Tiap Variabel Standar</b>")
    for title, body in [
        ("C1 – Visitor Count | trapmf[52,52,200,400] · trimf[200,425,650] · trapmf[500,650,799,799]",
         """<u>Rendah</u> (trapmf [52,52,200,400]):<br>
         <div class='formula-box'>μ_rendah(x) = 1 jika x≤200 (plateau) | (400−x)/200 jika 200&lt;x&lt;400 | 0 jika x≥400</div>
         <u>Sedang</u> (trimf [200,425,650]):<br>
         <div class='formula-box'>μ_sedang(x) = 0 jika x≤200 atau x≥650 | (x−200)/225 jika 200&lt;x≤425 | (650−x)/225 jika 425&lt;x&lt;650</div>
         <u>Tinggi</u> (trapmf [500,650,799,799]):<br>
         <div class='formula-box'>μ_tinggi(x) = 0 jika x≤500 | (x−500)/150 jika 500&lt;x≤650 | 1 jika x≥650 (plateau)</div>"""),
        ("C2 – Ticket Price | trapmf[10,10,30,55] · trimf[30,55,80] · trapmf[60,80,99,99]",
         """<u>Murah</u> (trapmf [10,10,30,55]):<br>
         <div class='formula-box'>μ_murah(x) = 1 jika x≤30 (plateau) | (55−x)/25 jika 30&lt;x&lt;55 | 0 jika x≥55</div>
         <u>Sedang</u> (trimf [30,55,80]):<br>
         <div class='formula-box'>μ_sedang(x) = 0 jika x≤30 atau x≥80 | (x−30)/25 jika 30&lt;x≤55 | (80−x)/25 jika 55&lt;x&lt;80</div>
         <u>Mahal</u> (trapmf [60,80,99,99]):<br>
         <div class='formula-box'>μ_mahal(x) = 0 jika x≤60 | (x−60)/20 jika 60&lt;x≤80 | 1 jika x≥80 (plateau)</div>"""),
        ("C3 – Satisfaction | trapmf[0,0,2,3.25] · trimf[2.5,3.5,4.5] · trapmf[3.75,4.5,5,5]",
         """<u>Rendah</u> (trapmf [0,0,2,3.25]):<br>
         <div class='formula-box'>μ_rendah(x) = 1 jika x≤2 (plateau) | (3.25−x)/1.25 jika 2&lt;x&lt;3.25 | 0 jika x≥3.25</div>
         <u>Sedang</u> (trimf [2.5,3.5,4.5]):<br>
         <div class='formula-box'>μ_sedang(x) = 0 jika x≤2.5 atau x≥4.5 | (x−2.5)/1 jika 2.5&lt;x≤3.5 | (4.5−x)/1 jika 3.5&lt;x&lt;4.5</div>
         <u>Tinggi</u> (trapmf [3.75,4.5,5,5]):<br>
         <div class='formula-box'>μ_tinggi(x) = 0 jika x≤3.75 | (x−3.75)/0.75 jika 3.75&lt;x≤4.5 | 1 jika x≥4.5 (plateau)</div>"""),
        ("C4 – Revenue Generated | trapmf[5k,5k,30k,55k] · trimf[30k,55k,80k] · trapmf[60k,80k,100k,100k]",
         """<u>Rendah</u> (trapmf [5000,5000,30000,55000]):<br>
         <div class='formula-box'>μ_rendah(x) = 1 jika x≤30000 (plateau) | (55000−x)/25000 jika 30000&lt;x&lt;55000 | 0 jika x≥55000</div>
         <u>Sedang</u> (trimf [30000,55000,80000]):<br>
         <div class='formula-box'>μ_sedang(x) = 0 jika x≤30k atau x≥80k | (x−30k)/25k jika 30k&lt;x≤55k | (80k−x)/25k jika 55k&lt;x&lt;80k</div>
         <u>Tinggi</u> (trapmf [60000,80000,100000,100000]):<br>
         <div class='formula-box'>μ_tinggi(x) = 0 jika x≤60000 | (x−60000)/20000 jika 60k&lt;x≤80k | 1 jika x≥80000 (plateau)</div>"""),
        ("C5 – Operational Cost | trapmf[2k,2k,15k,27k] · trimf[15k,27.5k,40k] · trapmf[30k,40k,50k,50k]",
         """<u>Rendah</u> (trapmf [2000,2000,15000,27000]):<br>
         <div class='formula-box'>μ_rendah(x) = 1 jika x≤15000 (plateau) | (27000−x)/12000 jika 15k&lt;x&lt;27k | 0 jika x≥27000</div>
         <u>Sedang</u> (trimf [15000,27500,40000]):<br>
         <div class='formula-box'>μ_sedang(x) = 0 jika x≤15k atau x≥40k | (x−15k)/12.5k jika 15k&lt;x≤27.5k | (40k−x)/12.5k jika 27.5k&lt;x&lt;40k</div>
         <u>Tinggi</u> (trapmf [30000,40000,50000,50000]):<br>
         <div class='formula-box'>μ_tinggi(x) = 0 jika x≤30000 | (x−30k)/10k jika 30k&lt;x≤40k | 1 jika x≥40000 (plateau)</div>"""),
        ("Output – Kinerja | trapmf[0,0,25,50] · trimf[25,50,75] · trapmf[50,75,100,100]",
         """<u>Rendah</u> (trapmf [0,0,25,50]):<br>
         <div class='formula-box'>μ_rendah(y) = 1 jika y≤25 (plateau) | (50−y)/25 jika 25&lt;y&lt;50 | 0 jika y≥50</div>
         <u>Sedang</u> (trimf [25,50,75]):<br>
         <div class='formula-box'>μ_sedang(y) = 0 jika y≤25 atau y≥75 | (y−25)/25 jika 25&lt;y≤50 | (75−y)/25 jika 50&lt;y&lt;75</div>
         <u>Tinggi</u> (trapmf [50,75,100,100]):<br>
         <div class='formula-box'>μ_tinggi(y) = 0 jika y≤50 | (y−50)/25 jika 50&lt;y≤75 | 1 jika y≥75 (plateau)</div>
         <br><b>Kategori output: Rendah y* &lt; 33.33 | Sedang 33.33 ≤ y* &lt; 66.67 | Tinggi y* ≥ 66.67</b>
         <br><b>Defuzzifikasi: y* = Σ[y·μ_agg(y)] / Σ[μ_agg(y)]</b>"""),
    ]:
        with st.expander(title):
            infobox(body)

    infobox("""
    <h4>Rule Base — 15 Aturan IF-THEN Standar</h4>
    <table style='width:100%;border-collapse:collapse;color:#FFF;'>
    <tr style='border-bottom:1px solid rgba(255,210,80,.4);'>
        <th style='padding:5px 10px;color:#FFD580;'>No.</th>
        <th style='padding:5px 10px;color:#FFD580;'>IF (Antecedent)</th>
        <th style='padding:5px 10px;color:#FFD580;'>THEN</th>
        <th style='padding:5px 10px;color:#FFD580;'>Justifikasi</th>
    </tr>""" + "".join(
        f"<tr><td style='padding:4px 10px;'>{r['no']}</td>"
        f"<td>" + " ∧ ".join(f"<b>{v.upper()} {t.upper()}</b>" for v,t in r['ant']) + "</td>"
        f"<td><b style='color:{label_color(r['cons'])};'>{r['cons'].upper()}</b></td>"
        f"<td style='font-size:.85rem;'>{r['alasan']}</td></tr>"
        for r in RULES_STANDAR
    ) + "</table>")

    er_t3 = st.session_state.get('custom_rules', [])
    if er_t3:
        infobox(f"""
        <h4>Rule Tambahan — {len(er_t3)} Aturan Custom</h4>
        <table style='width:100%;border-collapse:collapse;color:#FFF;'>
        <tr style='border-bottom:1px solid rgba(255,210,80,.4);'>
            <th style='padding:5px 10px;color:#FFD580;'>No.</th>
            <th style='padding:5px 10px;color:#FFD580;'>IF</th>
            <th style='padding:5px 10px;color:#FFD580;'>THEN</th>
            <th style='padding:5px 10px;color:#FFD580;'>Alasan</th>
        </tr>""" + "".join(
            f"<tr><td style='padding:4px 10px;'>CR{i+1}</td>"
            f"<td>{r.get('var_key','?').upper()} {r.get('var_term','?').upper()}</td>"
            f"<td><b style='color:{label_color(r.get('cons','rendah'))};'>{r.get('cons','?').upper()}</b></td>"
            f"<td>{r.get('alasan','—')}</td></tr>"
            for i,r in enumerate(er_t3)
        ) + "</table>")

# TAB 4 — KELOLA KRITERIA
with tab4:
    infobox("""
    <h4>Kelola Kriteria & Rule Tambahan</h4>
    Variabel baru yang ditambahkan akan <b>langsung masuk ke sistem fuzzy</b> sebagai antecedent nyata.
    """)

    st.markdown("### Tambah Variabel Fuzzy Baru")
    with st.form("form_var"):
        fa, fb = st.columns(2)
        with fa:
            nk = st.text_input("Kode variabel (huruf kecil, tanpa spasi)", placeholder="cth: aksesibilitas")
            nl = st.text_input("Label tampilan", placeholder="cth: C6 – Aksesibilitas")
            nt = st.selectbox("Tipe", ["Benefit (↑)", "Cost (↓)"])
            nu = st.text_input("Satuan", placeholder="cth: km, %")
        with fb:
            nlo = st.number_input("Universe Min", value=0.0)
            nhi = st.number_input("Universe Max", value=100.0)
            nst = st.number_input("Step", value=1.0, min_value=0.01)

        st.markdown("**Parameter MF (3 himpunan):**")
        fr, fs, ft = st.columns(3)
        with fr:
            st.markdown("**Rendah — trapmf [a,b,c,d]**")
            tr = st.text_input("Nama himpunan rendah", value="rendah", key="tr")
            ra,rb,rc,rd = st.number_input("a",value=0.0,key="ra"), st.number_input("b",value=0.0,key="rb"), st.number_input("c",value=30.0,key="rc"), st.number_input("d",value=50.0,key="rd")
        with fs:
            st.markdown("**Sedang — trimf [a,b,c]**")
            ts = st.text_input("Nama himpunan sedang", value="sedang", key="ts")
            sa,sb,sc = st.number_input("a",value=30.0,key="sa"), st.number_input("b",value=50.0,key="sb"), st.number_input("c",value=70.0,key="sc")
        with ft:
            st.markdown("**Tinggi — trapmf [a,b,c,d]**")
            tt = st.text_input("Nama himpunan tinggi", value="tinggi", key="tt")
            ta,tb,tc,td = st.number_input("a",value=60.0,key="ta"), st.number_input("b",value=80.0,key="tba"), st.number_input("c",value=100.0,key="tc"), st.number_input("d",value=100.0,key="td")

        if st.form_submit_button("➕ Tambah Variabel"):
            errs = []
            if not nk.strip() or ' ' in nk: errs.append("Kode tidak boleh kosong/mengandung spasi")
            if nk in VARS_STANDAR: errs.append(f"Kode '{nk}' sudah dipakai variabel standar")
            if any(cv['key']==nk for cv in st.session_state['custom_vars']): errs.append(f"Variabel '{nk}' sudah ada")
            if nhi <= nlo: errs.append("Max harus > Min")
            if not (nlo<=ra<=rb<=rc<=rd<=nhi): errs.append("trapmf Rendah: a≤b≤c≤d dalam range")
            if not (nlo<=sa<=sb<=sc<=nhi): errs.append("trimf Sedang: a≤b≤c dalam range")
            if not (nlo<=ta<=tb<=tc<=td<=nhi): errs.append("trapmf Tinggi: a≤b≤c≤d dalam range")

            if errs:
                for e in errs: st.error(f"⚠️ {e}")
            else:
                st.session_state['custom_vars'].append({
                    'key': nk.strip().lower(), 'label': nl or f"C{6+len(st.session_state['custom_vars'])}–{nk}",
                    'tipe': nt, 'unit': nu, 'lo': nlo, 'hi': nhi, 'step': nst,
                    'mf_type_rendah':'trapmf','term_rendah':tr,'mf_rendah':[ra,rb,rc,rd],
                    'mf_type_sedang':'trimf', 'term_sedang':ts,'mf_sedang':[sa,sb,sc],
                    'mf_type_tinggi':'trapmf','term_tinggi':tt,'mf_tinggi':[ta,tb,tc,td],
                })
                _fs_cache.clear(); st.success(f"✅ Variabel '{nk}' ditambahkan!"); st.rerun()

    st.markdown("---")
    st.markdown("### Tambah Rule IF-THEN")
    ek = [cv['key'] for cv in st.session_state['custom_vars']]
    if not ek:
        st.info("Tambahkan variabel fuzzy lebih dulu.")
    else:
        with st.form("form_rule"):
            rr1,rr2,rr3 = st.columns(3)
            with rr1: rv = st.selectbox("Variabel antecedent", ek)
            with rr2:
                scv = next((cv for cv in st.session_state['custom_vars'] if cv['key']==rv), None)
                ta_ = [scv['term_rendah'],scv['term_sedang'],scv['term_tinggi']] if scv else ['rendah','sedang','tinggi']
                rt = st.selectbox("Himpunan", ta_)
            with rr3: rc_ = st.selectbox("THEN Kinerja", ['rendah','sedang','tinggi'])
            ra_ = st.text_input("Justifikasi")

            if st.form_submit_button("✅ Tambah Rule"):
                st.session_state['custom_rules'].append({
                    'no': f"CR{len(st.session_state['custom_rules'])+1}",
                    'var_key': rv, 'var_term': rt, 'cons': rc_,
                    'ant': [(rv, rt)], 'alasan': ra_,
                })
                _fs_cache.clear(); st.success("✅ Rule ditambahkan!"); st.rerun()

    st.markdown("---")
    st.markdown("### Variabel & Rule Aktif")
    infobox(f"5 Kriteria Standar (C1–C5) tetap &nbsp;|&nbsp; Variabel Tambahan: <b>{len(st.session_state['custom_vars'])}</b> &nbsp;|&nbsp; Rule Tambahan: <b>{len(st.session_state['custom_rules'])}</b>")

    if st.session_state['custom_vars']:
        for idx, cv in enumerate(st.session_state['custom_vars']):
            ci, cd = st.columns([6,1])
            with ci:
                st.markdown(f"""<div class='custom-var-box'>
                <b>{cv['label']}</b> ({cv['tipe']}) | {cv['lo']}–{cv['hi']} {cv['unit']}<br>
                <small>{cv['term_rendah']}: trapmf{cv['mf_rendah']} · {cv['term_sedang']}: trimf{cv['mf_sedang']} · {cv['term_tinggi']}: trapmf{cv['mf_tinggi']}</small>
                </div>""", unsafe_allow_html=True)
            with cd:
                if st.button("🗑", key=f"dcv_{idx}"):
                    st.session_state['custom_rules'] = [r for r in st.session_state['custom_rules'] if r.get('var_key')!=cv['key']]
                    st.session_state['custom_vars'].pop(idx)
                    _fs_cache.clear(); st.rerun()

    if st.session_state['custom_rules']:
        for idx, r in enumerate(st.session_state['custom_rules']):
            ri, rd_ = st.columns([6,1])
            with ri:
                st.markdown(f"""<div class='custom-var-box'>
                <b>{r.get('no','CR')}</b>: IF {r.get('var_key','?').upper()} {r.get('var_term','?').upper()}
                → THEN Kinerja <b style='color:{label_color(r.get("cons","rendah"))};'>{r.get('cons','?').upper()}</b>
                — {r.get('alasan','—')}</div>""", unsafe_allow_html=True)
            with rd_:
                if st.button("🗑", key=f"dr_{idx}"):
                    st.session_state['custom_rules'].pop(idx); _fs_cache.clear(); st.rerun()

    if st.session_state['custom_vars'] or st.session_state['custom_rules']:
        if st.button("🗑 Reset Semua"):
            st.session_state['custom_vars'] = []; st.session_state['custom_rules'] = []
            _fs_cache.clear(); st.rerun()

    if st.session_state['custom_vars']:
        st.markdown("### Preview MF Variabel Tambahan")
        try:
            _, _, av_p = get_fuzzy_system(st.session_state['custom_vars'], [])
            cp = st.columns(min(len(st.session_state['custom_vars']), 3))
            for i, cv in enumerate(st.session_state['custom_vars']):
                if cv['key'] in av_p:
                    with cp[i%3]:
                        render_fig(fig_mf(av_p[cv['key']], cv['label']))
        except Exception as e: st.error(f"Error: {e}")

# TAB 5 — ANALISIS SENSITIVITAS
with tab5:
    infobox("""
    <h4>Analisis Sensitivitas — What-If Analysis</h4>
    Ubah satu variabel input, lihat pengaruhnya terhadap output kinerja (variabel lain tetap).<br>
    Plot di-cache — hanya dihitung ulang jika baseline berubah.
    """)

    st.markdown("**Nilai Baseline:**")
    s1,s2,s3 = st.columns(3)
    with s1:
        sv = st.slider("C1 Visitor", 52, 799, 418, 1, key="sv")
        sp = st.slider("C2 Price", 10, 99, 53, 1, key="sp")
    with s2:
        ss = st.slider("C3 Satisfaction", 0.0, 5.0, 3.7, 0.1, key="ss")
        sr = st.number_input("C4 Revenue", 5000, 100000, 52000, 1000, key="sr")
    with s3:
        so = st.number_input("C5 Op.Cost", 2000, 50000, 25000, 500, key="so")

    base_sa = {'visitor':sv,'price':sp,'satisfaction':ss,'revenue':sr,'op_cost':so}
    rb = hitung_kinerja(base_sa)
    if rb.get('nilai'):
        infobox(f"<b>Kinerja Baseline:</b> <b style='color:#FFD580;font-size:1.3rem;'>{rb['nilai']:.4f}</b> ({rb['label']})")

    VAR_CHOICES = {
        "C1 – Visitor Count": ("visitor", list(np.arange(52, 800, 10))),
        "C2 – Ticket Price": ("price", list(np.arange(10, 100, 2))),
        "C3 – Satisfaction": ("satisfaction", list(np.arange(0, 5.1, 0.1))),
        "C4 – Revenue": ("revenue", list(np.arange(5000, 100001, 2000))),
        "C5 – Op. Cost": ("op_cost", list(np.arange(2000, 50001, 1000))),
    }

    sel = st.selectbox("Variabel yang dianalisis:", list(VAR_CHOICES.keys()))

    if st.button("▶ Jalankan Analisis Sensitivitas"):
        vk, vr = VAR_CHOICES[sel]
        cfg = _cache_key(st.session_state.get('custom_vars',[]), st.session_state.get('custom_rules',[]))
        with st.spinner("Menghitung..."):
            fs, vals, outs = fig_sensitivity_cached(tuple(sorted(base_sa.items())), vk, tuple(vr), sel, cfg)
        render_fig(fs)

        kp, kritis = None, []
        for v, o in zip(vals, outs):
            k = "Tinggi" if o>=66.67 else ("Sedang" if o>=33.33 else "Rendah")
            if kp and k != kp: kritis.append({'Nilai Input':round(float(v),2),'Output Kinerja':round(o,4),'Perubahan Kategori':f"{kp} → {k}"})
            kp = k
        if kritis:
            infobox("<b>Titik Kritis Pergantian Kategori:</b>"); st.dataframe(pd.DataFrame(kritis), use_container_width=True)
        else:
            st.info("Tidak ada pergantian kategori pada range ini.")

        if outs:
            infobox(f"<b>Statistik {sel}:</b> Min: <b>{min(outs):.4f}</b> | Max: <b>{max(outs):.4f}</b> | Range: <b>{max(outs)-min(outs):.4f}</b> | Mean: <b>{np.mean(outs):.4f}</b>")

    st.markdown("---")
    if st.button("▶ Bandingkan Semua Variabel"):
        cfg = _cache_key(st.session_state.get('custom_vars',[]), st.session_state.get('custom_rules',[]))
        rows_sum = []
        with st.spinner("Menghitung semua variabel..."):
            for vn,(vk,vr) in VAR_CHOICES.items():
                _,_,outs_s = fig_sensitivity_cached(tuple(sorted(base_sa.items())), vk, tuple(vr), vn, cfg)
                if outs_s:
                    rng = max(outs_s)-min(outs_s)
                    rows_sum.append({"Variabel":vn,"Min":round(min(outs_s),4),"Max":round(max(outs_s),4),
                                      "Range":round(rng,4),"Mean":round(np.mean(outs_s),4),
                                      "Sensitivitas":"🔴 Tinggi" if rng>30 else ("🟡 Sedang" if rng>15 else "🟢 Rendah")})
        df_s = pd.DataFrame(rows_sum).sort_values('Range', ascending=False).reset_index(drop=True)
        st.dataframe(df_s, use_container_width=True)
        if not df_s.empty:
            infobox(f"Paling berpengaruh: <b>{df_s.iloc[0]['Variabel']}</b> (range={df_s.iloc[0]['Range']:.4f})<br>"
                    f"Paling stabil: <b>{df_s.iloc[-1]['Variabel']}</b> (range={df_s.iloc[-1]['Range']:.4f})")

        lb_bar = [r['Variabel'].split('–')[-1].strip() for r in rows_sum]
        vl_bar = [r['Range'] for r in rows_sum]
        fig_r, ax_r = plt.subplots(figsize=(10,5)); fig_r.patch.set_facecolor(DARK_BG); ax_r.set_facecolor(DARK_BG)
        ax_r.barh(lb_bar, vl_bar, color=['#e74c3c' if v>30 else '#f39c12' if v>15 else '#2ecc71' for v in vl_bar], edgecolor=GOLD)
        for i,v in enumerate(vl_bar): ax_r.text(v+0.3, i, f"{v:.2f}", va='center', color='white', fontsize=9)
        _style(ax_r, "Range Output per Variabel (semakin besar = semakin sensitif)")
        ax_r.set_xlabel("Range Output Kinerja", color='white'); plt.tight_layout(); render_fig(fig_r)

# TAB 6 — PROFIL KELOMPOK
with tab6:
    infobox("""
    <h4>Profil Kelompok</h4>
    <b>Proyek Akhir — Sistem Cerdas Pendukung Keputusan 2025/2026</b><br><br>
    <b>Judul:</b> SPK Penilaian Kinerja Destinasi Wisata Kerajinan (Handicraft Tourism)<br>
    <b>Metode:</b> Fuzzy Logic – Mamdani<br>
    <b>Dataset:</b> Rural Heritage Tourism Industry Chain Dataset (784 baris, Heritage_Type = Handicraft Center)<br><br>
    <hr style='border-color:rgba(255,210,80,0.3);margin:12px 0;'>
    <b>Anggota:</b><br>
    &nbsp;&nbsp;• Lathiva Safina Almasea &nbsp;— NIM 123240226<br>
    &nbsp;&nbsp;• Mutiara Rahmawati Zalsa — NIM 123240257<br><br>
    <b>Program Studi:</b> Informatika — UPN "Veteran" Yogyakarta
    """)

    infobox("""
    <h4>Ringkasan Teknis</h4>
    <b>Struktur Aplikasi:</b><br>
    &nbsp;&nbsp;📋 Dataset — eksplorasi dan statistik data<br>
    &nbsp;&nbsp;⚙️ Hitung SPK — input nilai → hasil → detail Mamdani 5 langkah (tiap langkah dalam expander)<br>
    &nbsp;&nbsp;📊 Fungsi Keanggotaan — grafik MF + rumus + tabel rule base<br>
    &nbsp;&nbsp;🛠 Kelola Kriteria — tambah variabel & rule custom<br>
    &nbsp;&nbsp;📈 Analisis Sensitivitas — what-if analysis per variabel<br><br>
    <b>Optimasi Performa:</b><br>
    &nbsp;&nbsp;• Cache Fuzzy System — ControlSystem dibangun sekali per konfigurasi<br>
    &nbsp;&nbsp;• Batch Simulation — satu Simulation reusable untuk 784 baris<br>
    &nbsp;&nbsp;• Cache Sensitivitas — plot hanya dihitung ulang jika baseline berubah<br>
    &nbsp;&nbsp;• plt.close() — mencegah memory leak dari figure menumpuk<br>
    &nbsp;&nbsp;• st.session_state — hasil perhitungan disimpan, expander bisa dibuka/tutup tanpa hitung ulang<br><br>
    <b>Alur Mamdani (5 tahap):</b><br>
    &nbsp;&nbsp;① Fuzzifikasi → ② Evaluasi Rule → ③ Clipping (Implication) → ④ Agregasi → ⑤ Defuzzifikasi (Centroid CoG)
    """)