"""
prisma.py - dimensões de um PRISMA (lados da base, altura h, volume V), com incertezas, a
partir de um ou mais registos de varrimento, comparadas com os valores reais (paquímetro).
Mesmo método do dimensoes.py (cilindro), com a forma trocada por um prisma.

Formas:
    box           prisma retangular vertical (paralelepípedo, "caixa"): comprimento a,
                  largura b, altura h   (por defeito)
    regular N     prisma regular vertical com N lados de comprimento s (3 = triângulo,
                  6 = hexágono), escolhido com --sides N

Utilização:
    python prisma.py box_H12.txt                                    # um varrimento
    python prisma.py box_H12.txt box_H155.txt                       # repetições e/ou outras alturas
    python prisma.py box_H12.txt --a 6.00 0.02 --b 4.00 0.02 --h 10.0 0.05 --beta 13 1.5
    python prisma.py tri_H12.txt --sides 3 --s 5.00 0.02 --h 8.0 0.05 --beta 13 1.5

    --a/--b/--s/--h  valores do paquímetro: valor e sigma, em cm (a ordem de a e b é indiferente:
                     ambos são ordenados do maior para o menor, como os ajustados)
    --beta           meio-ângulo do feixe do varrimento da vareta: valor e sigma, em graus
                     (fixa beta no ajuste; o seu sigma entra na parte de calibração)
    --fast           salta a parte de calibração (vista rápida; orçamento incompleto)
    --R / --H        substituem R e H de todos os registos (cm)

Cada registo pode ser colado diretamente do Monitor Série; de cada ficheiro é usado o último
varrimento completo, com R e H da sua linha "# Geometria". Precisa do scan3d.py na mesma pasta,
e de numpy, scipy e matplotlib. Uma corrida completa demora alguns minutos (cada fonte de
calibração, e cada ponto do perfil da altura, é um ajuste completo). Grava a figura
<primeiro_registo>_prisma.png (vista de cima nos eixos do prisma + resíduos).

Colunas de cada leitura: [posição do prato, ângulo do prato (graus), inclinação do servo
(graus), distância medida (cm)]. Referencial do laboratório: z vertical ao longo do eixo do
prato (z = 0 no prato), x horizontal do eixo do prato para o sensor, que está em (R, H).

Método (o que vai para o relatório):
  1. A distância de cada leitura é corrigida (offset e escala da calibração, e triângulo
     altifalante-microfone: d = sqrt(d_medida^2 - (sep/2)^2)).
  2. O objeto é um prisma vertical: o tamanho e a altura são comuns a todos os varrimentos;
     o centro (cx, cy) e o ângulo de rotação phi sobre o prato são ajustados separadamente
     para cada varrimento (o objeto é reposto de novo de cada vez). Para cada leitura o
     modelo prevê a distância ao ponto da superfície MAIS PRÓXIMO que está DENTRO do cone do
     feixe (meio-ângulo beta) - o HC-SR04 reporta o primeiro eco, que vem desse ponto.
     Primeiro ajusta-se cada varrimento sozinho (com várias rotações iniciais, porque o
     ângulo phi tem mínimos locais), depois todos em conjunto (mínimos quadrados robustos,
     perda soft_l1).
  3. Incerteza de cada resultado, somada em quadratura:
       fit (ajuste)  da matriz de covariância do ajuste, s^2 (J^T J)^-1
       scan-scan     desvio-padrão do resultado ajustado a cada varrimento sozinho (>= 2)
       calibration   variação do resultado quando cada valor de calibração é deslocado do
                     seu sigma, refazendo o ajuste ("um de cada vez", somadas em quadratura)
  4. z = |scan - real| / sqrt(sigma_scan^2 + sigma_real^2)  (z < 2 compatível, 2-3 tensão,
     > 3 há uma causa a encontrar).
  5. Se a altura não ficar determinada, faz-se um perfil de verosimilhança em h: fixa-se h
     numa grelha de valores, reajusta-se tudo o resto, e vê-se que alturas os dados permitem
     (Delta chi^2 < 1 a 1 sigma, < 4 a 2 sigma). Tipicamente só dá um limite inferior.

Nota de física: uma face plana perpendicular ao feixe devolve a sua distância exata, seja qual
for a largura do feixe (o ponto mais próximo da face é o pé da perpendicular, que está no eixo);
é nas arestas e nos cantos que a largura do feixe conta. Por isso o tamanho face-a-face de um
prisma fica em geral melhor determinado do que o diâmetro de um cilindro.

Limites conhecidos:
  - Se beta for ajustado, o tamanho e a largura do feixe podem compensar-se (correlação
    mostrada; com |rho| > 0.7 deve medir-se beta com a vareta e usar --beta). É menos grave
    do que no cilindro, pelo argumento das faces planas acima.
  - A altura só é medida se o topo ficou dentro da zona alcançada pelo cone. Se o topo está
    acima do que o sensor vê, h não é dado (e o volume também não); dá-se apenas o intervalo
    permitido pelo perfil em h. Para medir h: um varrimento com o sensor acima do topo
    (H ~ h + 5 cm), inclinado para baixo.
  - Uma caixa tem simetria: phi e phi + 180 graus são iguais, e phi + 90 graus troca a com b;
    por isso a e b são sempre dados ordenados (maior primeiro) e não "comprimento/largura"
    com significado físico fixo.
  - A grelha de pontos de superfície é finita (28 pontos por aresta): resolução de
    ~ (lado / 28), bem abaixo do ruído das leituras para os tamanhos usados.
"""

import argparse
import os

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")   # sem janela: a figura é apenas gravada em ficheiro
import matplotlib.pyplot as plt

from scan3d import parse_log

# ========================= SETTINGS (CONFIGURAÇÃO) =========================
# Valores de calibração (correção aplicada aos dados) e respetivo sigma (incerteza dessa
# correção, usada na parte de calibração do orçamento). Alterar com a vossa calibração.
CAL = {
    #  nome           valor  sigma
    "R":            (None,  0.10),   # cm, eixo do prato -> eixo do servo; None = do registo
    "H":            (None,  0.10),   # cm, altura do eixo do servo acima do prato; None = do registo
    "servo_offset": (0.0,   1.0),    # graus somados à inclinação registada (erro do zero do servo)
    "dist_offset":  (0.0,   0.05),   # cm, real - medida (0 se já estiver corrigido no Arduino)
    "pivot_front":  (0.0,   0.10),   # cm, transdutores à frente do eixo do servo (ao longo do feixe)
    "pivot_up":     (0.0,   0.10),   # cm, transdutores acima do eixo do servo (sensor nivelado)
    "beta":         (None,  2.0),    # graus, meio-ângulo do feixe, do varrimento da vareta.
                                     # None = ajustá-lo a partir do próprio varrimento
}
DIST_SCALE = 1.0          # adimensional: real = DIST_SCALE * medida + dist_offset (1 = sem erro de escala)
TR_SEPARATION = 2.6       # cm, distância entre altifalante (emissor) e microfone (recetor)

MAX_RADIUS = 7.0          # cm: leituras colocadas a mais do que isto do eixo do prato são fundo;
                          # aumentar para objetos cuja meia-diagonal se aproxime deste valor
Z_MIN = 1.0               # cm: pontos reconstruídos abaixo desta altura são ecos do prato

BETA_START = 12.0         # graus, valor inicial do meio-ângulo do feixe (quando é ajustado)
ROBUST_SCALE = 0.3        # cm, escala da perda robusta soft_l1: resíduos muito maiores contam menos
                          # (ecos espúrios); da ordem do ruído de uma leitura
H_UNCONSTRAINED = 1.0     # cm: se o sigma total de h exceder isto, a altura não é dada
SOFT_MIN = 0.03           # cm, suavização do "ponto mais próximo" (mínimo suave)
CONE_EDGE = 0.002         # suavização da borda do cone, em unidades de cos(ângulo)
# ============================================================================

# Frações fixas usadas para amostrar a superfície (escaladas pelo tamanho e pela altura, para
# que os pontos deslizem suavemente quando os parâmetros mudam)
_E = np.linspace(0, 1, 28, endpoint=False)   # amostras ao longo de cada aresta (sem repetir o canto)
_ZF = np.linspace(0, 1, 40)                  # amostras na vertical, em cada face lateral
_RF = np.linspace(0, 1, 10)                  # amostras do centro para fora, na face do topo


class Shape:
    """Geometria do prisma: caixa (retângulo a x b) ou prisma regular de N lados s.

    Atributos: box (True para caixa), N (número de lados da base), n_size (número de
    parâmetros de tamanho: 2 para a caixa, 1 para o regular), names (nomes desses
    parâmetros: ["a", "b"] ou ["s"]).
    """

    def __init__(self, sides, box):
        """sides: N do prisma regular (ignorado se box); box: True para paralelepípedo."""
        self.box = box
        self.N = 4 if box else sides
        self.n_size = 2 if self.box else 1
        self.names = ["a", "b"] if self.box else ["s"]

    def vertices(self, size, phi):
        """Vértices da base (array N x 2, cm), centrada na origem e rodada de phi (rad).

        size: [a, b] para a caixa, [s] para o regular. Caixa: a ao longo de x, b ao longo de y
        (com phi = 0). Regular: raio circunscrito Rc = s / (2 sin(pi/N)), vértices a ângulos
        2 pi k / N.
        """
        if self.box:
            a, b = size
            v = np.array([[a / 2, b / 2], [-a / 2, b / 2], [-a / 2, -b / 2], [a / 2, -b / 2]])
        else:
            s = size[0]
            Rc = s / (2 * np.sin(np.pi / self.N))
            t = 2 * np.pi * np.arange(self.N) / self.N
            v = np.c_[Rc * np.cos(t), Rc * np.sin(t)]
        # rotação de phi no sentido direto (vetores-linha: v @ M equivale a M^T v)
        c, sn = np.cos(phi), np.sin(phi)
        return v @ np.array([[c, sn], [-sn, c]])

    def surface(self, size, h, phi):
        """Nuvem de pontos da superfície visível (faces laterais + topo), array M x 3 em cm.

        Entradas: size (tamanho da base, cm), h (altura, cm), phi (rotação, rad). A base não
        entra porque está apoiada no prato.
        """
        v = self.vertices(size, phi)
        w = np.roll(v, -1, axis=0)                     # vértice seguinte de cada aresta
        # pontos ao longo de cada aresta: v + t (w - v), t em [0, 1)
        edge = v[:, None, :] + _E[None, :, None] * (w - v)[:, None, :]      # (N, E, 2)
        edge = edge.reshape(-1, 2)
        # faces laterais de z0 = 0.3 cm (o canto com o prato confunde-se com o prato) até h
        z0 = 0.3
        Z = z0 + _ZF * (h - z0)
        side = np.c_[np.repeat(edge, len(Z), axis=0), np.tile(Z, len(edge))]
        # topo: cópias do perímetro encolhidas para o centro (fator 0 a 1), em z = h; cobrem
        # a face do topo porque a base é convexa e centrada na origem
        top_xy = (_RF[:, None, None] * edge[None, :, :]).reshape(-1, 2)
        top = np.c_[top_xy, np.full(len(top_xy), h)]
        return np.vstack([side, top])

    def volume(self, size, h):
        """Volume (cm^3) = área da base x h. Caixa: a b h; regular: N s^2 / (4 tan(pi/N)) x h."""
        if self.box:
            return size[0] * size[1] * h
        s = size[0]
        return self.N * s**2 * h / (4 * np.tan(np.pi / self.N))

    def volume_rel_sigma(self, size, ssize, h, sh):
        """Incerteza relativa do volume, por propagação em quadratura (parâmetros supostos
        independentes). Caixa: sqrt((sa/a)^2 + (sb/b)^2 + (sh/h)^2); regular: V ~ s^2 h, logo
        sqrt((2 ss/s)^2 + (sh/h)^2)."""
        if self.box:
            return np.sqrt((ssize[0] / size[0]) ** 2 + (ssize[1] / size[1]) ** 2 + (sh / h) ** 2)
        return np.sqrt((2 * ssize[0] / size[0]) ** 2 + (sh / h) ** 2)


def load(path, R=None, H=None):
    """Lê um ficheiro de registo e devolve o último varrimento completo.

    Entradas: path (ficheiro .txt), R e H (cm) opcionais que substituem os do registo.
    Saída: dict com name, d (leituras válidas N x 4), R, H (cm) e n_total (leituras antes
    de retirar as inválidas).
    """
    scans = parse_log(path)
    # último varrimento completo; se nenhum terminou ("# Fim"), o último que existir
    complete = [s for s in scans if s["complete"]] or scans
    s = complete[-1]
    d = s["rows"]
    # distância 0 = sem eco (timeout): leitura descartada
    return dict(name=os.path.basename(path), d=d[d[:, 3] > 0],
                R=R or s["R"], H=H or s["H"], n_total=len(d))


def geometry(d, R, H, cal):
    """Posição do sensor, direção do feixe e distância corrigida de cada leitura (plano x-z do lab.).

    Entradas: d (leituras N x 4), R, H (cm), cal (calibração em uso).
    Saídas (arrays N): sx, sz (posição dos transdutores, cm), ux, uz (vetor unitário do eixo
    do feixe), dist (distância corrigida, cm).
    """
    # inclinação do servo, positiva = feixe virado para baixo
    inc = np.radians(d[:, 2] + cal["servo_offset"])
    # correção linear (escala e offset)
    dist = DIST_SCALE * d[:, 3] + cal["dist_offset"]
    # triângulo emissor-alvo-recetor: medida = sqrt(dist^2 + (sep/2)^2) -> tira-se dist
    # (np.maximum evita a raiz de um negativo para alvos muito próximos)
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))
    # eixo do feixe: para -x (para o eixo do prato), para baixo quando inc > 0
    ux, uz = -np.cos(inc), -np.sin(inc)
    # normal ao feixe, "para cima" no referencial do sensor
    nx, nz = -np.sin(inc), np.cos(inc)
    # braço do pivô: os transdutores estão pivot_front à frente e pivot_up acima do eixo do
    # servo, e esse braço roda com a inclinação
    sx = R + cal["pivot_front"] * ux + cal["pivot_up"] * nx
    sz = H + cal["pivot_front"] * uz + cal["pivot_up"] * nz
    return sx, sz, ux, uz, dist


def select_object(scan, cal):
    """Mantém as leituras que são do objeto: o ponto no eixo do feixe fica a menos de
    MAX_RADIUS do eixo do prato e acima do prato. (Um corte fixo na distância deitaria fora
    leituras reais do objeto quando o sensor está longe, p.ex. R = 14 cm.)

    Entradas: scan (dict de load), cal. Saída: cópia do dict com d reduzido.
    """
    d = scan["d"]
    sx, sz, ux, uz, dist = geometry(d, scan["R"], scan["H"], cal)
    # ponto reconstruído "ingénuo": à distância medida, sobre o eixo do feixe
    x, z = sx + dist * ux, sz + dist * uz
    return dict(scan, d=d[(np.abs(x) <= MAX_RADIUS) & (z >= Z_MIN)])


def predict(scan, geo, P, beta):
    """Distância suave ao ponto da superfície mais próximo dentro do cone, para cada leitura.

    Entradas: scan (usa os ângulos do prato), geo (saída de geometry), P (pontos da
    superfície M x 3, cm, já deslocados para o centro do objeto no referencial do prato),
    beta (graus).
    Saída: distância prevista (cm) por leitura. 'Mais próximo' (mínimo suave) e 'dentro'
    (borda suave do cone) são suavizados para o modelo variar continuamente com os
    parâmetros - senão o ajuste fica preso em patamares.
    """
    sx, sz, ux, uz, _ = geo
    cb = np.cos(np.radians(beta))
    ang = np.radians(scan["d"][:, 1])
    out = np.empty(len(ang))
    # agrupa as leituras pelo ângulo do prato (uma rotação da superfície por ângulo)
    for a in np.unique(ang):
        idx = np.where(ang == a)[0]
        # objeto rodado do ângulo do prato a; sensor fixo no semieixo +x (y = 0). Mesma
        # convenção de sinais da reconstrução: (x, 0) no lab. = (x cos a, -x sin a) no prato.
        x = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
        y = -(P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a))
        # vetor sensor -> ponto (linhas = leituras, colunas = pontos da superfície)
        vx = x[None, :] - sx[idx, None]
        vz = P[None, :, 2] - sz[idx, None]
        n = np.sqrt(vx**2 + y[None, :] ** 2 + vz**2)
        # cosseno do ângulo ao eixo do feixe; dentro do cone se c >= cos(beta)
        c = (vx * ux[idx, None] + vz * uz[idx, None]) / n
        # cone suave: softplus CONE_EDGE * log(1 + exp((cb - c)/CONE_EDGE)) ~ max(0, cb - c);
        # multiplicado por 200 cm empurra para longe os pontos fora do cone (~2 cm por 3 graus)
        n_eff = n + 200.0 * CONE_EDGE * np.logaddexp(0, (cb - c) / CONE_EDGE)
        # mínimo suave: média pesada por exp(-(n - n_min)/SOFT_MIN); tende para o mínimo
        # exato quando SOFT_MIN -> 0, mas sem saltos quando o ponto mais próximo muda.
        # Subtrair m evita overflow na exponencial.
        m = n_eff.min(axis=1, keepdims=True)
        w = np.exp(-(n_eff - m) / SOFT_MIN)
        out[idx] = (w * n_eff).sum(axis=1) / w.sum(axis=1)
    return out


def fit(scans, cal, shape, starts):
    """Ajuste do modelo do prisma a todas as leituras dos varrimentos dados.

    p = [tamanho..., h, beta, (cx, cy, phi) por varrimento]. beta fixo se cal['beta'] estiver
    definido. Tamanho, h, cx, cy em cm; beta em graus; phi em rad.

    Entradas: scans (lista de dicts), cal (calibração), shape (Shape), starts (lista de
    vetores p iniciais; fica a solução de menor custo).
    Saídas: p ajustado (vetor completo), sigma de cada parâmetro, resíduos (cm), geometrias
    e matriz de covariância completa.
    """
    geos = [geometry(s["d"], s["R"], s["H"], cal) for s in scans]
    ns, k0 = len(scans), shape.n_size
    npar = k0 + 2 + 3 * ns
    fixed_beta = cal.get("beta")
    # índices livres: todos menos beta (índice k0 + 1) quando beta está fixo
    free = [i for i in range(npar) if not (i == k0 + 1 and fixed_beta is not None)]

    def full(q):
        """Vetor completo p a partir dos parâmetros livres q (repõe beta fixo)."""
        p = np.empty(npar)
        p[free] = q
        if fixed_beta is not None:
            p[k0 + 1] = fixed_beta
        return p

    def residuals(q):
        """Resíduos previsto - medido (cm), de todas as leituras, concatenados."""
        p = full(q)
        size, h, beta = p[:k0], p[k0], p[k0 + 1]
        res = []
        for k, (s, g) in enumerate(zip(scans, geos)):
            # superfície com a rotação e o centro próprios deste varrimento
            cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
            P = shape.surface(size, h, phi)
            P[:, 0] += cx
            P[:, 1] += cy
            res.append(predict(s, g, P, beta) - g[4])
        return np.concatenate(res)

    # limites: tamanho 0.5-20 cm, h 1-25 cm, beta 3-40 graus, centro +-3 cm, phi +-2 pi
    lo = np.array([0.5] * k0 + [1.0, 3.0] + [-3.0, -3.0, -2 * np.pi] * ns)[free]
    hi = np.array([20.0] * k0 + [25.0, 40.0] + [3.0, 3.0, 2 * np.pi] * ns)[free]
    best = None
    for p0 in starts:
        # ponto inicial estritamente dentro dos limites
        q0 = np.clip(np.array(p0, float)[free], lo + 1e-3, hi - 1e-3)
        # perda robusta soft_l1: rho(z) = 2(sqrt(1+z) - 1), z = (res/f_scale)^2 -> quadrática
        # para resíduos pequenos, linear para grandes (outliers pesam pouco).
        # diff_step=0.01: passo relativo de 1% nas derivadas numéricas (maior do que o "grão"
        # da grelha de superfície, para o Jacobiano refletir a variação real do modelo)
        sol = least_squares(residuals, q0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=ROBUST_SCALE, diff_step=0.01)
        if best is None or sol.cost < best.cost:
            best = sol
    # covariância cov = s^2 (J^T J)^-1, com s^2 = soma(res^2)/(graus de liberdade): a
    # variância dos resíduos estimada dos próprios dados. J é o Jacobiano do least_squares,
    # já ponderado pela perda robusta.
    J, res = best.jac, best.fun
    s2 = np.sum(res**2) / max(len(res) - len(best.x), 1)
    try:
        cov_q = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        # J^T J singular: algum parâmetro não é determinado -> incerteza infinita
        cov_q = np.full((len(best.x), len(best.x)), np.inf)
    # covariância no vetor completo (beta fixo fica com linha/coluna a zero)
    cov = np.zeros((npar, npar))
    cov[np.ix_(free, free)] = cov_q
    return full(best.x), np.sqrt(np.abs(np.diag(cov))), res, geos, cov


def height_profile(scans, cal, shape, p, sigma_res):
    """Perfil de verosimilhança em h: refaz o ajuste com h fixo numa série de valores.

    Devolve as alturas que os dados permitem a 1 e a 2 sigma (Delta chi^2 < 1 e < 4). Usado
    quando h não fica determinado: tipicamente o topo nunca foi visto e só existe um limite
    inferior.

    Entradas: scans, cal, shape, p (melhor ajuste, ponto de partida), sigma_res (cm, sigma
    de uma leitura, estimado dos resíduos).
    Saídas: grid (alturas testadas, cm), dchi2 (Delta chi^2 em cada uma), e os intervalos
    (min, max) das alturas com Delta chi^2 < 1 e < 4.
    """
    geos = [geometry(s["d"], s["R"], s["H"], cal) for s in scans]
    k0, ns = shape.n_size, len(scans)
    fixed_beta = cal.get("beta")
    # grelha de alturas de 0.5 em 0.5 cm, de ~5 cm abaixo do sensor mais baixo (mínimo 1.5 cm)
    # até 6 cm acima do mais alto
    lo_h = max(1.5, min(s["H"] for s in scans) - 5)
    grid = np.arange(lo_h, max(s["H"] for s in scans) + 6.01, 0.5)

    def make_res(hfix):
        """Devolve a função de resíduos com h fixo em hfix (cm). Os parâmetros livres q são
        [tamanho..., (beta se não fixo), (cx, cy, phi) por varrimento]."""
        def residuals(q):
            """Resíduos previsto - medido (cm) com h = hfix."""
            size = q[:k0]
            beta = fixed_beta if fixed_beta is not None else q[k0]
            rest = q[k0 + (0 if fixed_beta is not None else 1):]
            out = []
            for k, (s, g) in enumerate(zip(scans, geos)):
                cx, cy, phi = rest[3 * k: 3 * k + 3]
                P = shape.surface(size, hfix, phi)
                P[:, 0] += cx
                P[:, 1] += cy
                out.append(predict(s, g, P, beta) - g[4])
            return np.concatenate(out)
        return residuals

    # ponto de partida e limites sem h (h é fixado pela grelha)
    q0 = list(p[:k0]) + ([] if fixed_beta is not None else [p[k0 + 1]]) + list(p[k0 + 2:])
    lo = [0.5] * k0 + ([] if fixed_beta is not None else [3.0]) + [-3.0, -3.0, -2 * np.pi] * ns
    hi = [20.0] * k0 + ([] if fixed_beta is not None else [40.0]) + [3.0, 3.0, 2 * np.pi] * ns
    costs = []
    for hfix in grid:
        # cada altura parte do melhor ajuste e reajusta todos os outros parâmetros
        q0c = np.clip(q0, np.array(lo) + 1e-3, np.array(hi) - 1e-3)
        sol = least_squares(make_res(hfix), q0c, bounds=(lo, hi), loss="soft_l1",
                            f_scale=ROBUST_SCALE, diff_step=0.01)
        costs.append(sol.cost)
    costs = np.array(costs)
    # custo do least_squares = 1/2 soma(rho); para resíduos pequenos soft_l1 ~ quadrática, logo
    # custo ~ 1/2 soma(res^2) e chi^2 = soma(res^2)/sigma^2 = 2 custo / sigma^2. Assim
    # Delta chi^2 = 2 Delta custo / sigma^2 (relativo ao mínimo da grelha). Com 1 parâmetro de
    # interesse (h), Delta chi^2 < 1 é o intervalo a 1 sigma e < 4 a 2 sigma.
    dchi2 = 2 * (costs - costs.min()) / sigma_res**2        # limite de resíduos pequenos da soft_l1
    ok1, ok2 = grid[dchi2 < 1], grid[dchi2 < 4]
    return grid, dchi2, (ok1.min(), ok1.max()), (ok2.min(), ok2.max())


def first_guess(scan, cal, shape):
    """Estimativa inicial a partir da nuvem bruta de um varrimento.

    Saídas: size (tamanho inicial, cm), h0 (altura inicial, cm: 1 cm acima do ponto mais
    alto visto) e phis (rotações iniciais a experimentar, rad).
    """
    sx, sz, ux, uz, dist = geometry(scan["d"], scan["R"], scan["H"], cal)
    # "raio" típico: 0.8 x mediana da distância ao eixo da nuvem bruta (que está inchada para
    # fora, porque o eco vem do ponto mais próximo no cone e não do eixo do feixe)
    r0 = 0.8 * np.median(np.abs(sx + dist * ux))
    h0 = np.max(sz + dist * uz) + 1.0
    if shape.box:
        # caixa ligeiramente retangular (b = 0.8 a) para quebrar a simetria a = b
        size = [1.7 * r0, 1.7 * r0 * 0.8]
        phis = np.radians([0, 30, 60])                    # caixa: phi + 90 graus = a e b trocados
    else:
        size = [2 * r0 * np.tan(np.pi / shape.N)]          # lado de um N-ágono de raio inscrito r0
        # rotações espaçadas dentro de um período de simetria (360/N graus)
        phis = np.radians(np.arange(3) * 120 / shape.N)
    return size, h0, phis


def results_from(p, shape):
    """Resultados físicos a partir de p: dict com size (tamanhos ordenados, maior primeiro,
    cm), h (cm) e V (cm^3). A ordenação resolve a ambiguidade a <-> b da caixa."""
    k0 = shape.n_size
    size = sorted(p[:k0], reverse=True)
    return dict(size=np.array(size), h=p[k0], V=shape.volume(size, p[k0]))


def main():
    """Lê os argumentos, ajusta cada varrimento e depois todos em conjunto, calcula o
    orçamento de incerteza, verifica se a altura foi medida (senão faz o perfil em h),
    imprime os resultados e a comparação com o paquímetro, e grava a figura."""
    ap = argparse.ArgumentParser(description="Prism dimensions with uncertainties from scan logs")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--sides", type=int, default=0, help="regular prism with N sides (default: box)")
    ap.add_argument("--a", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="box: caliper length (cm)")
    ap.add_argument("--b", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="box: caliper width (cm)")
    ap.add_argument("--s", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="regular: caliper side (cm)")
    ap.add_argument("--h", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper height (cm)")
    ap.add_argument("--R", type=float, help="override R for all logs (cm)")
    ap.add_argument("--H", type=float, help="override H for all logs (cm)")
    ap.add_argument("--beta", nargs=2, type=float, metavar=("VALUE", "SIGMA"),
                    help="beam half-angle from the rod sweep (deg)")
    ap.add_argument("--fast", action="store_true", help="skip the calibration part")
    args = ap.parse_args()

    # --sides 0 (por defeito) = caixa
    shape = Shape(args.sides, box=(args.sides == 0))
    kind = "box" if shape.box else f"regular prism, {shape.N} sides"
    k0 = shape.n_size

    # valores de calibração em uso (os sigmas ficam em CAL)
    cal = {k: v[0] for k, v in CAL.items()}
    if args.beta:
        # beta medido à parte: fixo no ajuste, sigma usado na parte de calibração
        CAL["beta"] = tuple(args.beta)
        cal["beta"] = args.beta[0]
    scans = [select_object(load(f, args.R or cal["R"], args.H or cal["H"]), cal) for f in args.logs]
    print(f"Shape: {kind}")
    for s in scans:
        print(f"{s['name']}: R={s['R']} H={s['H']}, {len(s['d'])} object readings of {s['n_total']}")

    # 1. cada varrimento sozinho (várias rotações iniciais, para não ficar num mínimo local
    #    de phi), depois todos em conjunto a partir da média dos parâmetros comuns e dos
    #    centros/rotações de cada um
    per, per_p = [], []
    for s in scans:
        size0, h0, phis = first_guess(s, cal, shape)
        starts = [list(size0) + [h0, BETA_START, 0.0, 0.0, ph] for ph in phis]
        pk = fit([s], cal, shape, starts)[0]
        per_p.append(pk)
        per.append(results_from(pk, shape))
    if len(scans) > 1:
        start = list(np.mean([q[:k0 + 2] for q in per_p], axis=0))
        for pk in per_p:
            start += list(pk[k0 + 2:k0 + 5])
        p, sp, res, geos, cov = fit(scans, cal, shape, [start])
    else:
        p, sp, res, geos, cov = fit(scans, cal, shape, [per_p[0]])
    best = results_from(p, shape)
    # as sigmas do ajuste têm de seguir a mesma ordenação (maior primeiro) dos tamanhos
    order = np.argsort(-p[:k0])   # ordenado (maior primeiro), como results_from
    sig = {n: {"fit": sp[:k0][order][i]} for i, n in enumerate(shape.names)}
    sig["h"] = {"fit": sp[k0]}

    # 2. varrimento-a-varrimento: desvio-padrão amostral (ddof=1) dos ajustes individuais
    if len(scans) >= 2:
        for i, n in enumerate(shape.names):
            sig[n]["scan-scan"] = np.std([q["size"][i] for q in per], ddof=1)
        sig["h"]["scan-scan"] = np.std([q["h"] for q in per], ddof=1)

    # 3. sensibilidades à calibração, "um de cada vez": desloca-se cada valor de +1 sigma,
    #    reajusta-se a partir da melhor solução e regista-se |Delta| de cada resultado
    contrib = {}
    if not args.fast:
        for name, (value, sigma) in CAL.items():
            # beta ajustado: a sua incerteza já está na covariância do ajuste
            if name == "beta" and cal["beta"] is None:
                continue
            cal2 = dict(cal)
            if name in ("R", "H"):
                # R e H pertencem a cada varrimento: desloca-se em todos
                scans2 = [dict(s, **{name: s[name] + sigma}) for s in scans]
            else:
                cal2[name] = (value or 0.0) + sigma
                scans2 = scans
            q = results_from(fit(scans2, cal2, shape, [list(p)])[0], shape)
            contrib[name] = [abs(q["size"][i] - best["size"][i]) for i in range(k0)] + [abs(q["h"] - best["h"])]
        # contribuição total da calibração = soma em quadratura das fontes
        for i, n in enumerate(shape.names + ["h"]):
            sig[n]["calibration"] = np.sqrt(sum(c[i] ** 2 for c in contrib.values()))

    # incerteza total = soma em quadratura das partes
    total = {n: np.sqrt(sum(v ** 2 for v in sig[n].values())) for n in sig}
    # a altura só conta como medida se o seu sigma for pequeno E o topo ajustado estiver
    # dentro do que o cone do feixe realmente alcançou (senão o topo nunca foi visto e h é
    # arbitrário). z_seen = altura máxima do bordo SUPERIOR do cone: esse bordo está
    # (inc - beta) abaixo da horizontal, com inc = arcsin(-uz), e sobe dist * sin(beta - inc).
    z_seen = max(np.max(g[1] + g[4] * np.sin(np.radians(p[k0 + 1]) - np.arcsin(-g[3]))) for g in geos)
    h_ok = total["h"] < H_UNCONSTRAINED and best["h"] <= z_seen + 0.5

    # tamanho da nuvem bruta (pontos colocados no eixo do feixe), para mostrar o enviesamento
    # que o modelo do cone corrige
    raw = []
    for k, (s, g) in enumerate(zip(scans, geos)):
        sx, sz, ux, uz, dist = g
        x = sx + dist * ux
        ang = np.radians(s["d"][:, 1])
        cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
        # para o referencial do prato (mesma convenção de predict), centrado no prisma
        xo, yo = x * np.cos(ang) - cx, -x * np.sin(ang) - cy
        c, sn = np.cos(-phi), np.sin(-phi)                  # rodar de -phi: eixos do próprio prisma
        raw.append(np.c_[xo * c - yo * sn, xo * sn + yo * c])
    raw = np.vstack(raw)
    if shape.box:
        # cada ponto é atribuído à face mais "próxima" em proporção: faces x = +-a/2 se
        # |x|/a > |y|/b, senão faces y = +-b/2; o lado bruto é 2 x mediana da coordenada
        a_, b_ = p[0], p[1]
        on_x = np.abs(raw[:, 0]) / a_ > np.abs(raw[:, 1]) / b_
        raw_size = sorted([2 * np.median(np.abs(raw[on_x, 0])), 2 * np.median(np.abs(raw[~on_x, 1]))], reverse=True)
    else:
        rin = np.median(np.hypot(raw[:, 0], raw[:, 1]))     # aproximadamente o raio inscrito
        raw_size = [2 * rin * np.tan(np.pi / shape.N)]

    # ---------------- relatório no ecrã ----------------
    print("\n=== Fit ===")
    # qualidade do ajuste: mediana de |resíduo| e fração de leituras a menos de 0.5 cm
    print(f"Readings fitted: {len(res)} | median |residual| {np.median(np.abs(res)):.2f} cm | "
          f"within 0.5 cm: {100 * np.mean(np.abs(res) < 0.5):.0f}%")
    if cal["beta"] is None:
        # correlação entre o primeiro tamanho e beta: rho = cov / (sigma sigma); |rho| ~ 1
        # significa que tamanho e largura do feixe se compensam (degenerescência)
        rho = cov[0, k0 + 1] / np.sqrt(cov[0, 0] * cov[k0 + 1, k0 + 1]) if cov[0, 0] * cov[k0 + 1, k0 + 1] > 0 else 0
        print(f"Beam half-angle beta = {p[k0 + 1]:.1f} ± {sp[k0 + 1]:.1f} deg (fitted; correlation with size: {rho:+.2f})")
        if abs(rho) > 0.7:
            print("  WARNING: size and beam width trade off. Measure beta with the rod sweep (--beta).")
    else:
        print(f"Beam half-angle beta = {p[k0 + 1]:.1f} ± {CAL['beta'][1]:.1f} deg (fixed, from the rod sweep)")
    # descentramento e rotação do objeto em cada varrimento
    for k, s in enumerate(scans):
        cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
        print(f"{s['name']}: centre offset {np.hypot(cx, cy):.2f} cm, rotation {np.degrees(phi) % 360:.0f} deg")

    # orçamento de incerteza: uma linha por parte, calibração discriminada por fonte
    names = shape.names + ["h"]
    print("\n=== Uncertainty budget (cm) ===")
    print(f"{'part':14s}" + "".join(f"{n:>10s}" for n in names))
    for part in sig["h"]:
        print(f"{part:14s}" + "".join(f"{sig[n].get(part, 0):10.2f}" for n in names))
    if contrib:
        print("  calibration, by source:")
        for name, c in contrib.items():
            print(f"    {name:12s}" + "".join(f"{x:10.2f}" for x in c))
    print(f"{'TOTAL':14s}" + "".join(f"{total[n]:10.2f}" for n in names))

    print("\n=== Result ===")
    for i, n in enumerate(shape.names):
        label = {"a": "Length a", "b": "Width  b", "s": "Side   s"}[n]
        print(f"{label} = {best['size'][i]:.2f} ± {total[n]:.2f} cm   (raw point cloud: {raw_size[i]:.2f} cm)")
    if h_ok:
        # volume com incerteza propagada em quadratura (ver Shape.volume_rel_sigma)
        ssize = [total[n] for n in shape.names]
        sV = best["V"] * shape.volume_rel_sigma(best["size"], ssize, best["h"], total["h"])
        print(f"Height   h = {best['h']:.2f} ± {total['h']:.2f} cm")
        print(f"Volume   V = {best['V']:.1f} ± {sV:.1f} cm^3")
    else:
        print("Height: not determined directly - checking which heights the data allow ...")
        # sigma de uma leitura estimado de forma robusta: MAD = mediana(|res - mediana(res)|);
        # para ruído gaussiano MAD = 0.6745 sigma, logo sigma = 1.4826 MAD. Ao contrário do
        # desvio-padrão, não é inflacionado pelos ecos espúrios.
        mad = np.median(np.abs(res - np.median(res)))
        grid, dchi2, b1, b2 = height_profile(scans, cal, shape, p, 1.4826 * mad)
        top_grid = grid.max()
        def bound(b):
            """Texto do intervalo (min, max) de h; um extremo que toca o limite da grelha
            significa que os dados não limitam h desse lado."""
            lo_txt = f"h >= {b[0]:.1f} cm" if b[0] > grid.min() else "no lower bound"
            hi_txt = f"h <= {b[1]:.1f} cm" if b[1] < top_grid else "no upper bound"
            return f"{lo_txt}, {hi_txt}"
        print(f"  1 sigma: {bound(b1)}")
        print(f"  2 sigma: {bound(b2)}")
        # tabela h : Delta chi^2 da grelha inteira
        print("  " + "  ".join(f"{g:.1f}:{d:.0f}" for g, d in zip(grid, dchi2)) + "   (h: delta chi^2)")
        if b1[1] >= top_grid:
            print("  The top was never seen (it is above what the sensor looks at): only a lower bound.")
            print("  To measure h, add a scan with the sensor above the top (H about h + 5 cm), tilted down.")
    # resultados de cada varrimento ajustado sozinho (base da parte scan-scan)
    if len(scans) > 1:
        for s, q in zip(scans, per):
            print(f"  alone, {s['name']}: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(shape.names, q["size"]))
                  + f", h={q['h']:.2f}")

    # comparação com o paquímetro; os valores reais de a e b são ordenados (maior primeiro)
    # tal como os ajustados, para comparar lado com lado
    real = []
    if shape.box and (args.a or args.b):
        real = sorted([x for x in (args.a, args.b) if x], key=lambda t: -t[0])
    elif not shape.box and args.s:
        real = [args.s]
    if real:
        print()
        for i, (val, sv) in enumerate(real):
            n = shape.names[i]
            # z-score do ajuste e da nuvem bruta (mesma sigma), incertezas em quadratura
            z = abs(best["size"][i] - val) / np.hypot(total[n], sv)
            zr = abs(raw_size[i] - val) / np.hypot(total[n], sv)
            print(f"vs caliper {n} = {val:.2f} ± {sv:.2f} cm: difference {best['size'][i] - val:+.2f} cm, z = {z:.1f}"
                  f"   (raw cloud: {raw_size[i] - val:+.2f} cm, z = {zr:.1f})")
    if args.h and h_ok:
        hr, shr = args.h
        z = abs(best["h"] - hr) / np.hypot(total["h"], shr)
        print(f"vs caliper h = {hr:.2f} ± {shr:.2f} cm: difference {best['h'] - hr:+.2f} cm, z = {z:.1f}")
    print("z < 2 consistent, 2-3 tension, > 3 find the cause")

    # ---------------- figura ----------------
    # à esquerda: vista de cima nos eixos do próprio prisma (nuvem bruta, contorno ajustado,
    # contorno do paquímetro); à direita: resíduos vs ângulo do prato
    base = os.path.splitext(args.logs[0])[0] + "_prisma"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    a1.scatter(raw[:, 0], raw[:, 1], s=6, alpha=0.5, c="#4c78a8", label="points (raw, on the beam axis)")
    # contorno ajustado com phi = 0 (já estamos nos eixos do prisma); vértice repetido para fechar
    v = shape.vertices(p[:k0], 0.0)
    v = np.vstack([v, v[:1]])
    a1.plot(v[:, 0], v[:, 1], "r-", lw=2, label="fit: " + ", ".join(
        f"{n} = {best['size'][i]:.2f} ± {total[n]:.2f}" for i, n in enumerate(shape.names)) + " cm")
    if real:
        # contorno do paquímetro alinhado com o ajustado: o lado maior vai para o mesmo eixo
        # em que o ajuste pôs o seu lado maior
        rs = [x[0] for x in real]
        if shape.box and len(rs) == 2:
            big_first = p[0] >= p[1]
            vr = shape.vertices(rs if big_first else rs[::-1], 0.0)
        else:
            vr = shape.vertices([rs[0]], 0.0)
        vr = np.vstack([vr, vr[:1]])
        a1.plot(vr[:, 0], vr[:, 1], "k--", lw=1.5, label="caliper")
    a1.plot(0, 0, "k+", ms=10)   # centro do prisma
    a1.set_aspect("equal")
    a1.set_xlabel("x along the prism (cm)"); a1.set_ylabel("y across the prism (cm)")
    a1.set_title("Top view, in the prism's own axes"); a1.legend(fontsize=8, loc="lower left")
    # resíduos de cada varrimento (fatias consecutivas do vetor res); devem estar centrados em 0
    # e sem tendência; faixa cinzenta = +-ROBUST_SCALE
    off = 0
    for s in scans:
        n = len(s["d"])
        a2.scatter(s["d"][:, 1], res[off:off + n], s=8, alpha=0.6, label=s["name"])
        off += n
    a2.axhline(0, color="k", lw=1)
    a2.axhspan(-ROBUST_SCALE, ROBUST_SCALE, color="gray", alpha=0.2)
    a2.set_xlabel("plate angle (deg)"); a2.set_ylabel("measured - model (cm)")
    a2.set_title("Residuals: no trend expected"); a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=130)
    print(f"\nFigure: {base}.png")


if __name__ == "__main__":
    main()
