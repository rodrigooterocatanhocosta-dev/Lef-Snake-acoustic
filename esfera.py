"""
esfera.py - raio de uma (meia) esfera a partir das leituras brutas do sensor de ultrassons.

O QUE FAZ
  Lê um registo (log) de varrimento, coloca cada leitura no referencial do objeto (posição do
  sensor S e direção do eixo do feixe U) e ajusta uma esfera (centro e raio) às distâncias medidas
  com um modelo EXATO de "primeiro eco".

  NOTA: os resultados finais do meio globo usam hemi.py / globo_resultados.py (meia esfera + prato).
  Este ficheiro mantém-se porque hemi.py usa as suas funções readings() e geometry().

COMO CORRER
  Usage: python esfera.py globo12.txt globo14.txt [--beta 12.4 1.9]
  (precisa de scan3d.py na mesma pasta; --beta = meia-abertura do feixe e a sua incerteza, em graus)

FÍSICA DO MODELO (exato para uma esfera; não é preciso adivinhar a forma do feixe)
  O HC-SR04 devolve o PRIMEIRO eco, isto é, a distância ao ponto do objeto mais próximo do sensor
  que esteja dentro do cone do feixe (meia-abertura beta). Para uma esfera de centro C e raio r,
  vista do sensor S à distância D = |S - C|, seja gamma o ângulo entre o eixo do feixe e a direção S->C:
     alpha = max(0, gamma - beta)              (direção do cone mais próxima do centro)
     d     = D cos(alpha) - sqrt(r^2 - D^2 sin^2(alpha))
  Porquê: a distância de S à superfície ao longo de uma direção que faz ângulo alpha com S->C é a
  menor raiz da interseção raio-esfera; ela cresce com alpha, logo o primeiro eco vem da direção do
  cone com menor alpha. Se o centro estiver dentro do cone (gamma <= beta), alpha = 0 e d = D - r,
  seja qual for beta. As leituras cujo cone falha a esfera (D sin(alpha) > r) não são usadas.

MÉTODO, PASSO A PASSO
  1. readings(): último varrimento completo do ficheiro; só leituras com eco (distância > 0).
  2. geometry(): para cada leitura, S e U no referencial do objeto (o prato rodou: roda-se de volta);
     corta ecos do prato (ponto no eixo abaixo de Z_MIN).
  3. fit(): mínimos quadrados robustos (perda soft_l1) dos parâmetros (cx, cy, cz, r).
  4. run(): 1.º ajuste -> retira leituras muito afastadas do modelo (outros ecos) -> 2.º ajuste;
     depois sensibilidade à calibração: perturba um parâmetro de cada vez em 1 sigma e reajusta.
  5. main(): repete com beta + sigma(beta) e soma todas as contribuições de calibração em quadratura.

SAÍDAS
  Texto no terminal, por ficheiro: r ± sigma(ajuste) ± sigma(calibração), altura e posição do centro,
  sigma dos resíduos e número de leituras usadas; depois a variação de r e de cz por cada parâmetro
  de calibração.

INCERTEZA
  Covariância do ajuste (+) calibração (reajustes um-de-cada-vez), em quadratura - a mesma receita de
  dimensoes.py / prisma.py. A dispersão entre varrimentos (R = 12 vs R = 14) vê-se comparando as
  linhas impressas para cada ficheiro.

LIMITES CONHECIDOS
  - Modela uma esfera COMPLETA; o meio globo tem face plana e está sobre o prato (ver hemi.py).
  - O feixe é tratado como um cone de bordo nítido (sem perfil de intensidade).
  - As incertezas de calibração em CAL são valores por defeito, não medidos.
"""
import argparse
import numpy as np
from scipy.optimize import least_squares

import scan3d

# Incertezas (1 sigma) dos parâmetros de calibração, usadas como perturbações nos reajustes:
#   R, H          posição do sensor (distância ao eixo e altura), cm
#   servo_offset  erro do zero do servo (inclinação), graus
#   dist_offset   desvio constante da distância medida, cm
#   pivot_front   distância do pivô do servo ao sensor ao longo do feixe, cm
#   pivot_up      idem, na perpendicular ao feixe, cm
CAL = {"R": 0.10, "H": 0.10, "servo_offset": 1.0, "dist_offset": 0.05,
       "pivot_front": 0.10, "pivot_up": 0.10}
Z_MIN = 1.0   # cm: leituras cujo ponto (no eixo do feixe) fica abaixo disto são ecos do prato


def readings(path):
    """Lê as leituras do último varrimento completo de um ficheiro de registo.

    Entrada: path - caminho do ficheiro de registo do Arduino.
    Saída:   (d, R, H)
       d - array (N, >=4), colunas [posição, ângulo do prato (°), inclinação (°), distância (cm), ...],
           só com as linhas em que houve eco (distância > 0; 0 = sem eco);
       R - distância horizontal do sensor ao eixo do prato (cm), lida do cabeçalho do varrimento;
       H - altura do sensor acima do prato (cm).
    Usa o último varrimento marcado como completo (os interrompidos são ignorados).
    """
    s = [x for x in scan3d.parse_log(path) if x["complete"]][-1]
    d = s["rows"]
    d = d[d[:, 3] > 0]
    return d, s["R"], s["H"]


def geometry(d, R, H, cal):
    """Posição do sensor e direção do feixe, no referencial do objeto, para cada leitura.

    Entradas:
       d    - leituras (saída de readings());
       R, H - posição nominal do sensor (cm);
       cal  - dicionário de PERTURBAÇÕES de calibração (somadas aos valores nominais), por exemplo
              {"R": 0.1}; {} = valores nominais. É assim que se fazem os reajustes de calibração.
    Saídas (só leituras acima de Z_MIN):
       S    - (N, 3) posição do sensor (cm);
       U    - (N, 3) vetor unitário do eixo do feixe;
       dist - (N,) distância medida, já corrigida (cm).
    Referencial do objeto: z vertical a partir do prato, origem no eixo de rotação.
    """
    ang = np.radians(d[:, 1])
    inc = np.radians(d[:, 2] + cal.get("servo_offset", 0.0))
    dist = d[:, 3] + cal.get("dist_offset", 0.0)
    # Correção do triângulo: emissor e recetor estão separados de TR_SEPARATION; o som percorre
    # emissor -> ponto -> recetor, por isso a distância medida é a hipotenusa. A distância
    # perpendicular ao ponto médio entre os dois é sqrt(d^2 - (s/2)^2).
    dist = np.sqrt(np.maximum(dist**2 - (scan3d.TR_SEPARATION / 2) ** 2, 0))
    R = R + cal.get("R", 0.0)
    H = H + cal.get("H", 0.0)
    pf, pu = cal.get("pivot_front", 0.0), cal.get("pivot_up", 0.0)
    # Plano vertical x-z do braço: o feixe aponta para o eixo (-x) inclinado para baixo de 'inc'.
    # u = direção do feixe; n = a sua normal "para cima". O sensor não está no pivô do servo:
    # está pf à frente (ao longo de u) e pu acima (ao longo de n), por isso move-se com a inclinação.
    ux, uz = -np.cos(inc), -np.sin(inc)
    nx, nz = -np.sin(inc), np.cos(inc)
    sx = R + pf * ux + pu * nx
    sz = H + pf * uz + pu * nz
    # O prato rodou de 'ang'; no referencial do objeto é o sensor que roda de -ang em torno de z.
    c, s = np.cos(ang), -np.sin(ang)
    S = np.c_[sx * c, sx * s, sz]                 # sensor no referencial do objeto
    U = np.c_[ux * c, ux * s, uz]                 # eixo do feixe no referencial do objeto
    P = S + dist[:, None] * U                     # ponto colocado no eixo (só para o corte em z)
    # Leituras cujo ponto fica quase à altura do prato são ecos do prato, não do objeto.
    keep = P[:, 2] >= Z_MIN
    return S[keep], U[keep], dist[keep]


def model(p, S, U, beta):
    """Distância do primeiro eco prevista para uma esfera completa.

    Entradas:
       p    - (cx, cy, cz, r): centro e raio da esfera (cm);
       S, U - posições do sensor e eixos do feixe (de geometry());
       beta - meia-abertura do cone do feixe (graus).
    Saídas:
       pred - (N,) distância prevista d = D cos(alpha) - sqrt(r^2 - D^2 sin^2(alpha)) (cm);
       hit  - (N,) booleano: True se o cone toca a esfera (caso contrário pred não tem significado).
    """
    cx, cy, cz, r = p
    v = np.array([cx, cy, cz]) - S
    D = np.linalg.norm(v, axis=1)
    # gamma = ângulo entre o eixo do feixe e a direção sensor -> centro (clip evita arccos(1+eps)).
    cosg = np.clip((v * U).sum(1) / D, -1, 1)
    gam = np.arccos(cosg)
    # alpha = direção do cone mais próxima do centro; 0 se o centro já estiver dentro do cone.
    a = np.maximum(0.0, gam - np.radians(beta))
    # Discriminante da interseção raio-esfera: D sin(alpha) é a distância do centro a esse raio.
    # Se for > r o raio (e portanto todo o cone) passa ao lado da esfera: hit = False.
    disc = r**2 - (D * np.sin(a)) ** 2
    hit = disc > 0
    # Menor raiz = ponto de entrada na esfera = primeiro eco. O 1e-9 evita sqrt de negativos
    # nas leituras sem interseção (essas são tratadas à parte através de 'hit').
    pred = D * np.cos(a) - np.sqrt(np.maximum(disc, 1e-9))
    return pred, hit


def fit(S, U, dist, beta, p0):
    """Ajuste robusto da esfera às distâncias medidas.

    Entradas: S, U, dist (de geometry()), beta (graus), p0 = estimativa inicial (cx, cy, cz, r).
    Saídas:  (p, e, sig, nhit, n)
       p    - parâmetros ajustados (cm);
       e    - incertezas 1 sigma do ajuste, da covariância (cm);
       sig  - dispersão robusta dos resíduos (MAD), em cm: incerteza de UMA leitura;
       nhit - número de leituras cujo cone toca a esfera;  n - número total de leituras.
    """
    def res(p):
        """Resíduos (previsto - medido), em cm, para os parâmetros p."""
        pred, hit = model(p, S, U, beta)
        r = pred - dist
        r[~hit] = 3.0       # uma leitura cujo cone falha a esfera conta como um desajuste fixo de 3 cm
        return r
    # Perda soft_l1: rho(z) = 2(sqrt(1+z) - 1), com z = (res/f_scale)^2. Para |res| < 0.3 cm é
    # quadrática (mínimos quadrados normais); para resíduos grandes cresce só linearmente, por isso
    # ecos espúrios (outros objetos, prato, ecos múltiplos) puxam muito menos o ajuste.
    # f_scale = 0.3 cm é da ordem da resolução do sensor.
    sol = least_squares(res, p0, loss="soft_l1", f_scale=0.3)
    J = sol.jac
    r = res(sol.x)
    n, k = len(r), len(sol.x)
    # sigma de uma leitura pela MAD (desvio absoluto mediano): 1.4826*MAD = desvio padrão para
    # dados gaussianos, mas não é inflacionado pelos ecos espúrios (ao contrário do desvio padrão).
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    # Covariância dos parâmetros: cov = sigma^2 (J^T J)^-1 (aproximação de Gauss-Newton).
    # Usa-se a pseudo-inversa (pinv) para não falhar se algum parâmetro estiver mal determinado
    # (J^T J quase singular); nesse caso o seu sigma sai grande, como deve.
    cov = sig**2 * np.linalg.pinv(J.T @ J)
    pred, hit = model(sol.x, S, U, beta)
    return sol.x, np.sqrt(np.diag(cov)), sig, int(hit.sum()), n


def run(path, beta, p0=(0, 0, 0, 7)):
    """Análise completa de um ficheiro para um dado beta.

    Entradas: path (registo), beta (graus), p0 (estimativa inicial; por defeito centro no eixo,
              no prato, r = 7 cm).
    Saída: dicionário com
       p   - (cx, cy, cz, r) ajustados;  e - sigmas do ajuste;  sig - sigma dos resíduos;
       n   - leituras usadas;  ntot - leituras do último ajuste;  R - distância do sensor;
       cal - {parâmetro: variação de (cx, cy, cz, r) quando esse parâmetro muda 1 sigma}.
    """
    d, R, H = readings(path)
    S, U, dist = geometry(d, R, H, {})
    # Ciclo exterior: retira leituras muito afastadas do modelo ajustado (outros ecos).
    # Corte a 3 sigma, com um mínimo de 0.33 cm para sigma (não cortar demais se o ajuste for muito bom).
    p, e, sig, nh, n = fit(S, U, dist, beta, np.array(p0, float))
    pred, hit = model(p, S, U, beta)
    good = hit & (np.abs(pred - dist) < 3 * max(sig, 0.33))
    p, e, sig, nh, n = fit(S[good], U[good], dist[good], beta, p)
    # Sensibilidades à calibração (método um-de-cada-vez): desloca UM parâmetro de calibração do
    # seu sigma, recalcula a geometria, reajusta e regista a variação dos resultados. Supõe que os
    # erros de calibração são independentes e pequenos (resposta linear), por isso somam-se depois
    # em quadratura. beta não é tratado aqui (salta-se): é feito em main() com um run() completo.
    calvar = {}
    for k, sg in list(CAL.items()) + [("beta", None)]:
        if k == "beta":
            continue
        S2, U2, d2 = geometry(d, R, H, {k: sg})
        pr, h2 = model(p, S2, U2, beta)
        g2 = h2 & (np.abs(pr - d2) < 3 * max(sig, 0.33))
        p2 = fit(S2[g2], U2[g2], d2[g2], beta, p)[0]
        calvar[k] = p2 - p
    return dict(p=p, e=e, sig=sig, n=int(good.sum()), ntot=n, cal=calvar, R=R)


def main():
    """Ponto de entrada: analisa cada ficheiro dado na linha de comandos e imprime os resultados.

    Para cada ficheiro: run() com beta nominal, run() com beta + sigma(beta) (sensibilidade ao
    feixe), e incerteza de calibração = soma em quadratura de todas as variações.
    Saída: lista de dicionários de run() (com 'calsig' acrescentado).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--beta", nargs=2, type=float, default=[12.4, 1.9])
    args = ap.parse_args()
    beta, sbeta = args.beta
    out = []
    for f in args.logs:
        o = run(f, beta)
        # Sensibilidade a beta: reajuste completo com beta deslocado de 1 sigma.
        ob = run(f, beta + sbeta, o["p"])
        o["cal"]["beta"] = ob["p"] - o["p"]
        # sigma de calibração de cada parâmetro = raiz da soma dos quadrados das variações.
        cal = np.sqrt(sum(v**2 for v in o["cal"].values()))
        o["calsig"] = cal
        out.append(o)
        cx, cy, cz, r = o["p"]
        print(f"{f}: R={o['R']:.0f}  r = {r:.3f} +- {o['e'][3]:.3f} (fit) +- {cal[3]:.3f} (cal)   "
              f"centre z = {cz:.2f} +- {o['e'][2]:.2f}  centre xy = ({cx:.2f}, {cy:.2f})  "
              f"residual sigma {o['sig']:.2f} cm, {o['n']}/{o['ntot']} readings")
        for k, v in o["cal"].items():
            print(f"      cal {k:13s} dr = {v[3]:+.3f}  dcz = {v[2]:+.3f}")
    return out


if __name__ == "__main__":
    main()
