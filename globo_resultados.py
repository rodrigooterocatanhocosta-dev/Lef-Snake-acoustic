"""globo_resultados.py - meio globo (cúpula para cima, face plana no prato): altura do topo e raio.

O QUE FAZ
  Para os varrimentos globo12.txt (R = 12 cm) e globo14.txt (R = 14 cm) ajusta o modelo de hemi.py
  (primeiro eco dentro do cone do feixe, por traçado de raios) e tira duas estimativas do raio:
  A) esfera livre (altura do centro cz e raio a livres): o topo z_top = cz + a fica bem medido;
     a curvatura não (só se vê a calote acima de z ~ 5 cm, e cz e a estão fortemente correlacionados:
     uma esfera maior com centro mais baixo dá quase a mesma calote). Para uma meia esfera assente no
     prato, a = z_top  ->  r = z_top. Usa-se a forma conhecida e mede-se só uma grandeza (a altura).
  B) esfera obrigada a assentar no prato (cz = 0): r vem da curvatura da calote.
  sigma(z_top): perfil de verosimilhança (Delta chi^2 = 1) (+) reajustes de calibração um-de-cada-vez.

COMO CORRER
  python globo_resultados.py      (na pasta com globo12.txt, globo14.txt, hemi.py, esfera.py, scan3d.py)

MÉTODO, PASSO A PASSO (para cada ficheiro)
  1. Ajuste livre (orient 'up') -> p = (cx, cy, cz, a); T0 = cz + a.
  2. Perfil em z_top: reajusta com o topo FIXO em T0 - 0.3, T0, T0 + 0.3 cm (cz = T - a) e regista o
     custo mínimo em cada caso. A curvatura da parábola custo(T) dá sigma(z_top) (ver abaixo).
  3. Calibração: reajusta com R, H, servo, desvio de distância, pivô (+1 sigma cada) e com beta + 1.9°;
     a variação de z_top em cada caso soma-se em quadratura -> s_cal.
  4. sigma total = s_fit (+) s_cal.
  5. Método B: ajuste com cz = 0 fixo; raio e sigma só do ajuste.

PORQUÊ O PERFIL E NÃO A COVARIÂNCIA
  z_top é uma combinação (cz + a) de dois parâmetros muito correlacionados; o perfil mede
  diretamente quanto o ajuste piora quando o topo se afasta do ótimo, deixando os restantes
  parâmetros reajustarem-se. Com chi^2 = sum(r^2)/sig^2 e custo = (1/2) sum(r^2) (zona quadrática
  da soft_l1), Delta chi^2 = 2 Delta custo / sig^2. Se custo ~ custo0 + curv (T - T0)^2, então
  Delta chi^2 = 1 quando (T - T0)^2 = sig^2 / (2 curv): s_fit = sqrt(sig^2 / 2 / curv).

SAÍDAS
  Uma linha por ficheiro no terminal e o ficheiro globo_resultados.json (lido por build_excel.py) com:
  R, ztop, s_fit, s_cal, s_tot, a_free, cz_free, sig, n, rB, srB, sigB, cal (variações por parâmetro).

LIMITES CONHECIDOS
  - r = z_top supõe que a face plana assenta diretamente no prato (meia esfera exata).
  - O método B dá 9-10 cm nos dados reais, enquanto em simulações (dados sintéticos de uma cúpula de
    7 cm) dá 7,0: os dados reais NÃO são consistentes com uma cúpula de 7 cm vista por este modelo.
    A causa não foi identificada; fica registado como limite (a calote medida é mais plana).
  - As incertezas não incluem o enviesamento do próprio modelo (cone de bordo nítido, 97 raios).
"""
import json, numpy as np, hemi
# Incertezas de calibração (1 sigma) usadas nos reajustes um-de-cada-vez (cm, ou graus no servo).
CAL = [("R", .1), ("H", .1), ("servo_offset", 1.0), ("dist_offset", .05), ("pivot_front", .1), ("pivot_up", .1)]
out = {}
for f, R in [("globo12.txt", 12), ("globo14.txt", 14)]:
    # A) Ajuste livre da cúpula: p = (cx, cy, cz, a); topo T0 = cz + a.
    p, e, sig, cost, n = hemi.fit(f, "up")
    T0 = p[2] + p[3]
    # Perfil de verosimilhança: três ajustes com o topo fixo em T (cz = T - a, o resto livre).
    # 'T=T' no lambda congela o valor de T de cada iteração (sem isso todos usariam o último T).
    # [3] = custo final do ajuste.
    Ts = T0 + np.array([-0.3, 0, 0.3])
    cs = [hemi.fit(f, "up", fix={"cz": (lambda pp, T=T: T - pp[3])}, p0=[p[0], p[1], T - p[3], p[3]])[3] for T in Ts]
    # Segunda diferença finita / h^2 = d2(custo)/dT2; dividida por 2 dá o coeficiente curv da parábola.
    curv = (cs[0] - 2 * cs[1] + cs[2]) / 0.3**2 / 2          # custo ~ curv (T-T0)^2
    # Delta chi^2 = 2 Delta custo / sig^2 = 1  ->  sigma(z_top) = sqrt(sig^2 / (2 curv)).
    s_fit = np.sqrt(sig**2 / 2 / curv)
    # Calibração: variação de z_top quando cada parâmetro de calibração muda 1 sigma (partindo de p).
    dc = {}
    for k, sg in CAL:
        q = hemi.fit(f, "up", p0=p, cal={k: sg})[0]; dc[k] = q[2] + q[3] - T0
    # beta: 12.4 + 1.9° (sigma da calibração do feixe na caixa).
    q = hemi.fit(f, "up", beta=12.4 + 1.9, p0=p)[0]; dc["beta"] = q[2] + q[3] - T0
    # Soma em quadratura (erros de calibração supostos independentes).
    s_cal = float(np.sqrt(sum(v**2 for v in dc.values())))
    # B) Esfera com o centro forçado no prato (cz = 0): raio pela curvatura da calote.
    pB, eB, sigB, costB, _ = hemi.fit(f, "up", fix={"cz": 0.0})
    # s_tot = s_fit (+) s_cal em quadratura (np.hypot).
    out[f] = dict(R=R, ztop=float(T0), s_fit=float(s_fit), s_cal=s_cal, s_tot=float(np.hypot(s_fit, s_cal)),
                  a_free=float(p[3]), cz_free=float(p[2]), sig=float(sig), n=n,
                  rB=float(pB[3]), srB=float(eB[3]), sigB=float(sigB), cal=dc)
    print(f, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[f].items() if k != "cal"})
json.dump(out, open("globo_resultados.json", "w"), indent=1)
