"""
dimensoes.py - dimensões de um objeto CILÍNDRICO (diâmetro D, altura h, volume V), com
incertezas, a partir de um ou mais registos de varrimento, comparadas com os valores reais
(medidos com paquímetro).

Utilização:
    python dimensoes.py scan_a.txt                         # um varrimento
    python dimensoes.py scan_a.txt scan_b.txt scan_c.txt   # repetições e/ou outras alturas
    python dimensoes.py scan_a.txt --D 2.60 0.02 --h 11.0 0.05
                                   (diâmetro e altura do paquímetro, cada um valor e sigma, cm)

    Opções:
      --D VALOR SIGMA     diâmetro medido com o paquímetro (cm) -> calcula o z-score
      --h VALOR SIGMA     altura medida com o paquímetro (cm)   -> calcula o z-score
      --R / --H           substituem R e H de todos os registos (cm)
      --beta VALOR SIGMA  meio-ângulo do feixe medido no varrimento da vareta (graus);
                          fixa beta no ajuste em vez de o ajustar (recomendado, ver limites)
      --fast              salta a parte de calibração (só para uma vista rápida; o
                          orçamento de incerteza fica incompleto)

Cada registo pode ser colado diretamente do Monitor Série do Arduino; de cada ficheiro é usado
o último varrimento completo, com R e H lidos da sua linha "# Geometria". Precisa do scan3d.py
na mesma pasta, e de numpy, scipy e matplotlib. Produz no ecrã o ajuste, o orçamento de
incerteza e o resultado, e grava a figura <primeiro_registo>_dimensoes.png.

Colunas de cada leitura (linha do registo): [posição do prato, ângulo do prato (graus),
inclinação do servo (graus), distância medida (cm)].

Referencial do laboratório: eixo z vertical, ao longo do eixo de rotação do prato (z = 0 no
prato); eixo x horizontal, do eixo do prato para o sensor. O sensor (eixo do servo) está em
(x, z) = (R, H) e o feixe aponta para o objeto, inclinado para baixo de um ângulo "inc".

Método (o que vai para o relatório):
  1. A distância de cada leitura é corrigida: desvio (offset) e escala da calibração, e o
     triângulo altifalante-microfone (o som vai do emissor ao alvo e volta ao recetor, que
     estão separados de TR_SEPARATION; a distância perpendicular é sqrt(d^2 - (sep/2)^2)).
  2. O objeto é modelado como um cilindro vertical (raio r, altura h), com um centro (cx, cy)
     próprio para cada varrimento (o objeto nunca fica exatamente no eixo do prato, e é
     reposto entre varrimentos). Para cada leitura o modelo prevê a distância ao ponto da
     superfície MAIS PRÓXIMO que está DENTRO do cone do feixe (meio-ângulo beta): é isso que o
     HC-SR04 mede, porque reporta o primeiro eco que chega acima do limiar, e o primeiro eco
     vem do ponto mais próximo que o feixe ilumina - não necessariamente do eixo do feixe.
     r, h, beta e os centros são ajustados a todas as leituras em conjunto, por mínimos
     quadrados robustos (perda soft_l1, para que ecos espúrios pesem pouco).
  3. Incerteza de cada resultado, somada em quadratura (fontes supostas independentes):
       fit (ajuste)   da matriz de covariância do ajuste, s^2 (J^T J)^-1
       scan-scan      desvio-padrão do resultado ajustado a cada varrimento sozinho (>= 2
                      varrimentos): mede a reprodutibilidade (reposição do objeto, ruído, ...)
       calibration    variação do resultado quando cada valor de calibração é deslocado do
                      seu sigma (R, H, offset do servo, offset da distância, pivô), refazendo
                      o ajuste de cada vez ("um de cada vez"); as variações somam-se em
                      quadratura
  4. Comparação com o paquímetro: z = |scan - real| / sqrt(sigma_scan^2 + sigma_real^2);
     z < 2 compatível, 2-3 tensão, > 3 há uma causa a encontrar.
     Também se mostra o diâmetro da "nuvem bruta" (pontos colocados no eixo do feixe, sem
     modelo), para mostrar o enviesamento que o modelo do cone corrige.

Limites conhecidos:
  - Degenerescência raio-beta: num cilindro, um feixe mais largo e um raio menor dão
    praticamente as mesmas distâncias "mais próximas" que um feixe mais estreito e um raio
    maior (a superfície curva entra no cone mais cedo ou mais tarde). Se beta for ajustado,
    o programa mostra a correlação r-beta; com |rho| > 0.7 o diâmetro não é fiável e beta
    deve ser medido à parte (varrimento da vareta) e fixado com --beta. Para objetos largos
    a curvatura é pequena e o problema diminui; para objetos finos é grave.
  - A altura só é medida se o topo do objeto ficou dentro da zona que o cone do feixe
    alcançou. Se o topo está acima do que o feixe vê, os dados só dão um limite inferior e
    h fica arbitrário: o programa declara a altura "NOT constrained" e não dá h nem V.
    Solução: juntar um varrimento com o sensor acima do topo, inclinado para baixo.
  - A soma em quadratura das contribuições de calibração supõe-nas independentes e lineares
    (deslocamento de +1 sigma apenas); o volume ignora a correlação entre D e h.
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
# Valores de calibração e respetivo sigma (cm ou graus). "Valor" = correção que se aplica
# aos dados; "sigma" = incerteza dessa correção, usada na parte de calibração do orçamento.
# Alterar quando houver uma calibração própria (ex.: offset da distância medido contra uma
# régua, beta medido com o varrimento da vareta).
CAL = {
    #  nome           valor  sigma
    "R":            (None,  0.10),   # cm, distância horizontal eixo do prato - eixo do servo.
                                     # None = lida do registo
    "H":            (None,  0.10),   # cm, altura do eixo do servo acima do prato. None = registo
    "servo_offset": (0.0,   1.0),    # graus somados à inclinação registada (erro do zero do servo)
    "dist_offset":  (0.0,   0.05),   # cm, real - medida (0 se já estiver corrigido no Arduino)
    "pivot_front":  (0.0,   0.10),   # cm, transdutores à frente do eixo do servo (ao longo do feixe)
    "pivot_up":     (0.0,   0.10),   # cm, transdutores acima do eixo do servo (com o sensor nivelado)
    "beta":         (None,  2.0),    # graus, meio-ângulo do feixe, do varrimento da vareta.
                                     # None = ajustá-lo a partir do próprio varrimento (só é de
                                     # confiança para objetos largos: num objeto fino, a largura
                                     # do objeto e a largura do feixe compensam-se uma à outra)
}
DIST_SCALE = 1.0          # adimensional: real = DIST_SCALE * medida + dist_offset (1 = sem erro de escala,
                          # p.ex. velocidade do som correta)
TR_SEPARATION = 2.6       # cm, distância entre os centros do altifalante (emissor) e do microfone (recetor)

# Que leituras são do objeto
MAX_RADIUS = 7.0          # cm: leituras colocadas a mais do que isto do eixo do prato são fundo (paredes,
                          # suporte); aumentar para objetos com mais de ~14 cm de largura
Z_MIN = 1.0               # cm: pontos reconstruídos abaixo desta altura são ecos do prato

# Ajuste
BETA_START = 12.0         # graus, meio-ângulo do feixe usado como valor inicial quando beta é ajustado
ROBUST_SCALE = 0.3        # cm, escala da perda robusta: resíduos muito maiores do que isto contam menos
                          # (ecos espúrios); da ordem do ruído típico de uma leitura
H_UNCONSTRAINED = 1.0     # cm: se o sigma total de h exceder isto, a altura é dada como não medida
# ============================================================================


def load(path, R=None, H=None):
    """Lê um ficheiro de registo e devolve o último varrimento completo.

    Entradas: path (caminho do ficheiro .txt), R e H (cm) opcionais que substituem os da
    linha "# Geometria" do registo (None = usar os do registo).
    Saída: dict com name (nome do ficheiro), d (array N x 4 das leituras válidas: posição,
    ângulo do prato em graus, inclinação em graus, distância em cm), R, H (cm) e n_total
    (número de leituras antes de retirar as inválidas).
    """
    scans = parse_log(path)
    # usa o último varrimento completo; se nenhum terminou ("# Fim"), usa o último que existir
    complete = [s for s in scans if s["complete"]] or scans
    s = complete[-1]
    d = s["rows"]
    # distância 0 = sem eco (timeout do HC-SR04): essas leituras são descartadas
    return dict(name=os.path.basename(path), d=d[d[:, 3] > 0],
                R=R or s["R"], H=H or s["H"], n_total=len(d))


def geometry(d, R, H, cal):
    """Posição do sensor, direção do feixe e distância corrigida de cada leitura (plano x-z do lab.).

    Entradas: d (leituras N x 4), R e H (cm, posição do eixo do servo), cal (dict com os
    valores de calibração em uso).
    Saídas (arrays de N elementos): sx, sz = posição dos transdutores (cm); ux, uz = vetor
    unitário do eixo do feixe; dist = distância corrigida ao alvo (cm).
    O plano x-z é o plano vertical que contém o eixo do prato e o sensor (y = 0).
    """
    # inclinação do servo, positiva = feixe virado para baixo
    inc = np.radians(d[:, 2] + cal["servo_offset"])
    # correção linear da distância (escala e offset da calibração)
    dist = DIST_SCALE * d[:, 3] + cal["dist_offset"]
    # triângulo emissor-alvo-recetor: o HC-SR04 mede metade do percurso emissor->alvo->recetor,
    # ou seja a hipotenusa sqrt(dist^2 + (sep/2)^2); tira-se a distância perpendicular.
    # np.maximum(..., 0) evita a raiz de um número negativo em alvos muito próximos.
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))
    # eixo do feixe: aponta para -x (para o eixo do prato) e para baixo quando inc > 0
    ux, uz = -np.cos(inc), -np.sin(inc)
    # normal ao feixe, "para cima" no referencial do sensor (vertical quando inc = 0)
    nx, nz = -np.sin(inc), np.cos(inc)
    # os transdutores não estão no eixo do servo: estão pivot_front à frente (ao longo do
    # feixe) e pivot_up acima; esse braço roda com a inclinação
    sx = R + cal["pivot_front"] * ux + cal["pivot_up"] * nx
    sz = H + cal["pivot_front"] * uz + cal["pivot_up"] * nz
    return sx, sz, ux, uz, dist


# Grelha fixa num cilindro unitário, escalada por (r, h): a superfície move-se de forma suave
# com r e h (os pontos deslizam em vez de aparecerem/desaparecerem, o que daria saltos no
# modelo e derivadas nulas ou infinitas no ajuste).
_TH = np.linspace(0, 2 * np.pi, 180, endpoint=False)   # 180 ângulos à volta (de 2 em 2 graus)
_T, _ZF = np.meshgrid(_TH, np.linspace(0, 1, 60))      # parede lateral: 60 frações de altura
_T2, _RF = np.meshgrid(_TH, np.linspace(0, 1, 14))     # tampa do topo: 14 frações do raio
SOFT_MIN = 0.03           # cm: suavização do "ponto mais próximo" (evita que o ajuste fique preso)
CONE_EDGE = 0.002         # suavização da borda do cone, em unidades de cos(ângulo)


def cylinder_surface(r, h):
    """Nuvem de pontos da superfície visível de um cilindro centrado na origem.

    Entradas: r (raio, cm), h (altura, cm).
    Saída: array M x 3 (x, y, z em cm) com a parede lateral (de z0 = 0.3 cm até h) e a
    tampa do topo (discos concêntricos em z = h). A base não entra: está apoiada no prato.
    """
    # a parede começa 0.3 cm acima do prato: o canto parede-prato é confundido com o prato
    # (e as leituras com z < Z_MIN já foram removidas como ecos do prato)
    z0 = 0.3
    side = np.c_[r * np.cos(_T).ravel(), r * np.sin(_T).ravel(), (z0 + _ZF * (h - z0)).ravel()]
    top = np.c_[(r * _RF * np.cos(_T2)).ravel(), (r * _RF * np.sin(_T2)).ravel(), np.full(_T2.size, h)]
    return np.vstack([side, top])


def predict(scan, geo, r, h, beta, cx, cy):
    """Distância ao ponto da superfície mais próximo dentro do cone, para cada leitura.

    'Mais próximo' e 'dentro' são suavizados (mínimo suave, borda do cone suave) para que o
    resultado varie continuamente com os parâmetros; caso contrário o ajuste fica preso.

    Entradas: scan (dict de load/select_object; usa os ângulos do prato), geo (saída de
    geometry), r, h (cm), beta (graus, meio-ângulo do cone), cx, cy (cm, centro do cilindro
    no referencial do prato).
    Saída: array com a distância prevista (cm) para cada leitura, a comparar com geo[4].
    """
    sx, sz, ux, uz, _ = geo
    # superfície do cilindro, deslocada para o seu centro no prato
    P = cylinder_surface(r, h)
    P[:, 0] += cx
    P[:, 1] += cy
    cb = np.cos(np.radians(beta))
    ang = np.radians(scan["d"][:, 1])
    out = np.empty(len(ang))
    # agrupa as leituras pelo ângulo do prato: para cada ângulo, roda-se a superfície uma só
    # vez e tratam-se todas as inclinações desse ângulo em bloco (vetorizado)
    for a in np.unique(ang):
        idx = np.where(ang == a)[0]
        # objeto rodado do ângulo do prato a; sensor fixo no semieixo +x (y = 0).
        # A convenção de sinais é a mesma usada para reconstruir os pontos: um ponto do lab.
        # em (x, 0) corresponde a (x cos a, -x sin a) no referencial do objeto.
        x = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
        y = -(P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a))
        # vetor sensor -> ponto da superfície, para cada leitura (linha) e ponto (coluna)
        vx = x[None, :] - sx[idx, None]
        vz = P[None, :, 2] - sz[idx, None]
        n = np.sqrt(vx**2 + y[None, :] ** 2 + vz**2)   # distância sensor-ponto (cm)
        # c = cosseno do ângulo entre o eixo do feixe e a direção do ponto (o eixo do feixe
        # não tem componente y); o ponto está dentro do cone se c >= cos(beta)
        c = (vx * ux[idx, None] + vz * uz[idx, None]) / n
        # cone suave: fora do cone a distância é aumentada de forma suave (cerca de 2 cm por
        # cada 3 graus fora, com beta ~ 12 graus). CONE_EDGE * logaddexp(0, u/CONE_EDGE) é um
        # "softplus": ~0 dentro do cone e ~(cb - c) fora, mas com derivada contínua. O fator
        # 200 cm torna os pontos bem fora do cone efetivamente invisíveis para o mínimo.
        outside = CONE_EDGE * np.logaddexp(0, (cb - c) / CONE_EDGE)
        n_eff = n + 200.0 * outside
        # mínimo suave: média das distâncias pesada por exp(-(n - n_min)/SOFT_MIN). Os pontos
        # a menos de ~SOFT_MIN do mínimo dominam; o resultado tende para o mínimo exato quando
        # SOFT_MIN -> 0, mas varia suavemente quando o ponto mais próximo muda de um ponto da
        # grelha para outro. Subtrair m evita overflow/underflow na exponencial.
        m = n_eff.min(axis=1, keepdims=True)
        w = np.exp(-(n_eff - m) / SOFT_MIN)            # média pesada dos pontos mais próximos
        out[idx] = (w * n_eff).sum(axis=1) / w.sum(axis=1)
    return out


def select_object(scan, cal):
    """Mantém as leituras que são do objeto: o ponto no eixo do feixe fica a menos de
    MAX_RADIUS do eixo do prato e acima do prato. (Um corte fixo na distância deitaria fora
    leituras reais do objeto quando o sensor está longe, p.ex. R = 14 cm.)

    Entradas: scan (dict de load), cal (calibração).
    Saída: cópia do dict com d reduzido às leituras do objeto.
    """
    d = scan["d"]
    sx, sz, ux, uz, dist = geometry(d, scan["R"], scan["H"], cal)
    # ponto reconstruído "ingénuo": à distância medida, sobre o eixo do feixe
    x, z = sx + dist * ux, sz + dist * uz
    return dict(scan, d=d[(np.abs(x) <= MAX_RADIUS) & (z >= Z_MIN)])


def fit(scans, cal, start=None):
    """Ajuste conjunto do modelo do cilindro a todas as leituras de todos os varrimentos.

    Parâmetros p = [r, h, beta, cx1, cy1, cx2, cy2, ...]; beta fica fixo se cal['beta'] estiver
    definido. r, h, cx, cy em cm; beta em graus. r, h e beta são comuns a todos os varrimentos;
    há um centro (cx, cy) por varrimento.

    Entradas: scans (lista de dicts de select_object), cal (calibração), start (lista p
    inicial; None = estimar a partir da nuvem bruta, com dois pontos de partida).
    Saídas: p (parâmetros ajustados, vetor completo), sigma de cada parâmetro (raiz da
    diagonal da covariância; 0 para beta fixo), resíduos (medida - modelo, cm), lista de
    geometrias (uma por varrimento) e a matriz de covariância completa.
    """
    geos = [geometry(s["d"], s["R"], s["H"], cal) for s in scans]
    ns = len(scans)
    fixed_beta = cal.get("beta")
    # índices de p que são livres no ajuste (beta, índice 2, só entra se não estiver fixo)
    free = [0, 1] + ([] if fixed_beta is not None else [2]) + list(range(3, 3 + 2 * ns))

    def full(q):
        """Reconstrói o vetor completo p a partir do vetor q dos parâmetros livres
        (repõe beta fixo, se for o caso)."""
        p = np.empty(3 + 2 * ns)
        p[free] = q
        if fixed_beta is not None:
            p[2] = fixed_beta
        return p

    def residuals(q):
        """Resíduos (cm) previsto - medido, para todas as leituras de todos os varrimentos,
        concatenados; é isto que least_squares minimiza."""
        p = full(q)
        r, h, beta = p[:3]
        res = []
        for k, (s, g) in enumerate(zip(scans, geos)):
            cx, cy = p[3 + 2 * k], p[4 + 2 * k]
            res.append(predict(s, g, r, h, beta, cx, cy) - g[4])
        return np.concatenate(res)

    if start is None:
        # estimativa inicial a partir da nuvem bruta (pontos colocados no eixo do feixe)
        raw_r, zmax = [], []
        for s, g in zip(scans, geos):
            sx, sz, ux, uz, dist = g
            raw_r.append(np.abs(sx + dist * ux))
            zmax.append(sz + dist * uz)
        # a nuvem bruta exagera o raio (o eco vem do ponto mais próximo no cone, não do eixo),
        # daí o fator 0.8; o topo começa 1 cm acima do ponto mais alto visto
        r0 = 0.8 * np.median(np.concatenate(raw_r))
        h0 = np.max(np.concatenate(zmax)) + 1.0
        # dois pontos de partida diferentes (raio menor/feixe mais largo no segundo), para
        # reduzir o risco de cair num mínimo local; fica o de menor custo
        starts = [[r0, h0, BETA_START], [0.6 * r0 / 0.8, h0 + 2, BETA_START + 5]]
        starts = [s0 + [0.0, 0.0] * ns for s0 in starts]   # centros começam no eixo do prato
    else:
        starts = [start]

    # limites físicos: r 0.3-10 cm, h 1-25 cm, beta 3-40 graus, centros a +-3 cm do eixo
    lo = np.array([0.3, 1.0, 3.0] + [-3.0] * 2 * ns)[free]
    hi = np.array([10.0, 25.0, 40.0] + [3.0] * 2 * ns)[free]
    best = None
    for p0 in starts:
        # o ponto inicial tem de estar estritamente dentro dos limites
        q0 = np.clip(np.array(p0, float)[free], lo + 1e-3, hi - 1e-3)
        # perda robusta soft_l1: rho(z) = 2(sqrt(1+z) - 1), com z = (res/f_scale)^2. Para
        # |res| << ROBUST_SCALE é quadrática (mínimos quadrados normais); para |res| >>
        # ROBUST_SCALE cresce só linearmente, logo um eco espúrio (outlier) puxa pouco.
        # diff_step=0.01: passo relativo (1%) das derivadas numéricas, suficientemente grande
        # para "ver" a variação do modelo suavizado e não o ruído de discretização da grelha.
        sol = least_squares(residuals, q0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=ROBUST_SCALE, diff_step=0.01)
        if best is None or sol.cost < best.cost:
            best = sol

    # covariância a partir do Jacobiano, escalada pela dispersão dos resíduos:
    # cov = s^2 (J^T J)^-1, com s^2 = soma(res^2) / (N_leituras - N_parâmetros), a variância
    # dos resíduos estimada dos próprios dados (não se supõe um sigma por leitura conhecido).
    # J é o Jacobiano devolvido pelo least_squares, já ponderado pela perda robusta, pelo que
    # os outliers contribuem pouco também aqui.
    J = best.jac
    res = best.fun
    dof = max(len(res) - len(best.x), 1)
    s2 = np.sum(res**2) / dof
    try:
        cov_q = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        # J^T J singular: algum parâmetro não é determinado pelos dados -> incerteza infinita
        cov_q = np.full((len(best.x), len(best.x)), np.inf)
    # covariância no vetor completo p (linhas/colunas de beta fixo ficam a zero)
    cov = np.zeros((3 + 2 * ns, 3 + 2 * ns))
    cov[np.ix_(free, free)] = cov_q
    # abs() protege de valores diagonais ligeiramente negativos por arredondamento numérico
    return full(best.x), np.sqrt(np.abs(np.diag(cov))), res, geos, cov


def results_from(p):
    """Converte o vetor de parâmetros p nos resultados físicos.

    Saída: dict com D = 2r (cm), h (cm), V = pi r^2 h (cm^3) e beta (graus).
    """
    r, h = p[0], p[1]
    return dict(D=2 * r, h=h, V=np.pi * r**2 * h, beta=p[2])


def main():
    """Lê os argumentos, faz o ajuste conjunto, calcula o orçamento de incerteza (ajuste,
    varrimento-a-varrimento, calibração), imprime os resultados e a comparação com o
    paquímetro, e grava a figura (perfil lateral + resíduos)."""
    ap = argparse.ArgumentParser(description="Object dimensions with uncertainties from scan logs")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--D", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper diameter (cm)")
    ap.add_argument("--h", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper height (cm)")
    ap.add_argument("--R", type=float, help="override R for all logs (cm)")
    ap.add_argument("--H", type=float, help="override H for all logs (cm)")
    ap.add_argument("--beta", nargs=2, type=float, metavar=("VALUE", "SIGMA"),
                    help="beam half-angle from the rod sweep (deg); overrides CAL['beta']")
    ap.add_argument("--fast", action="store_true", help="skip the calibration part (quick look)")
    args = ap.parse_args()

    # valores de calibração em uso (só o valor; os sigmas ficam em CAL)
    cal = {k: v[0] for k, v in CAL.items()}
    if args.beta:
        # beta medido à parte: fica fixo no ajuste e o seu sigma entra na parte de calibração
        CAL["beta"] = tuple(args.beta)
        cal["beta"] = args.beta[0]
    scans_all = [load(f, args.R or cal["R"], args.H or cal["H"]) for f in args.logs]
    scans = [select_object(s, cal) for s in scans_all]
    for s in scans:
        print(f"{s['name']}: R={s['R']} H={s['H']}, {len(s['d'])} object readings "
              f"of {s['n_total']}")

    # 1. ajuste conjunto de todos os varrimentos: dá os valores centrais e a parte "fit"
    p, sp, res, geos, cov = fit(scans, cal)
    best = results_from(p)
    # sigma(D) = 2 sigma(r), porque D = 2r
    sig = {"D": {"fit": 2 * sp[0]}, "h": {"fit": sp[1]}}

    # estimativa da nuvem bruta, sem modelo: 2 x mediana da distância horizontal ao eixo do
    # prato dos pontos colocados no eixo do feixe. Serve para mostrar o enviesamento: o eco
    # vem do ponto mais próximo dentro do cone, por isso a distância medida é mais curta do
    # que a distância ao longo do eixo e a nuvem bruta fica "inchada" para fora.
    raw_r = np.concatenate([np.abs(g[0] + g[4] * g[2]) for g in geos])
    D_raw = 2 * np.median(raw_r)

    # 2. varrimento-a-varrimento (reprodutibilidade): cada varrimento ajustado sozinho, a
    # partir da solução conjunta; o desvio-padrão amostral (ddof=1) dos resultados é a
    # contribuição "scan-scan" (com só 2-3 varrimentos é uma estimativa grosseira)
    if len(scans) >= 2:
        per = []
        for k, s in enumerate(scans):
            start = list(p[:3]) + list(p[3 + 2 * k: 5 + 2 * k])
            pk = fit([s], cal, start=start)[0]
            per.append(results_from(pk))
        sig["D"]["scan-scan"] = np.std([q["D"] for q in per], ddof=1)
        sig["h"]["scan-scan"] = np.std([q["h"] for q in per], ddof=1)
    else:
        per = None

    # 3. sensibilidade à calibração, "um de cada vez": desloca-se cada valor de calibração
    # de +1 sigma, refaz-se o ajuste a partir da melhor solução, e a variação |Delta D|,
    # |Delta h| é a contribuição dessa fonte. As fontes somam-se em quadratura.
    contrib = {}
    if not args.fast:
        for name, (value, sigma) in CAL.items():
            cal2 = dict(cal)
            if name == "beta" and cal["beta"] is None:
                continue                      # beta ajustado: a sua incerteza e correlação
                                              # com r já estão na covariância do ajuste
            if name in ("R", "H"):
                # R e H pertencem a cada varrimento (vêm do registo): desloca-se em todos
                scans2 = [dict(s, **{name: s[name] + sigma}) for s in scans]
            else:
                cal2[name] = (value or 0.0) + sigma
                scans2 = scans
            p2 = fit(scans2, cal2, start=list(p))[0]
            q = results_from(p2)
            contrib[name] = (abs(q["D"] - best["D"]), abs(q["h"] - best["h"]))
        sig["D"]["calibration"] = np.sqrt(sum(c[0] ** 2 for c in contrib.values()))
        sig["h"]["calibration"] = np.sqrt(sum(c[1] ** 2 for c in contrib.values()))

    # incerteza total = soma em quadratura das partes (fit, scan-scan, calibration)
    total = {k: np.sqrt(sum(v ** 2 for v in sig[k].values())) for k in sig}
    # a altura só conta como medida se o seu sigma for pequeno E o topo ajustado estiver
    # dentro do que o cone do feixe realmente alcançou; senão o topo nunca foi visto e h é
    # arbitrário. z_seen = altura máxima atingida pelo bordo SUPERIOR do cone: esse bordo
    # faz um ângulo (inc - beta) abaixo da horizontal, com inc = arcsin(-uz), por isso sobe
    # dist * sin(beta - inc) acima do sensor. Tolerância de 0.5 cm.
    z_seen = max(np.max(g[1] + g[4] * np.sin(np.radians(p[2]) - np.arcsin(-g[3]))) for g in geos)
    h_ok = total["h"] < H_UNCONSTRAINED and best["h"] <= z_seen + 0.5

    # ---------------- relatório no ecrã ----------------
    # (o texto impresso fica em inglês, tal como no original)
    print("\n=== Fit ===")
    # qualidade do ajuste: mediana de |resíduo| e fração de leituras a menos de 0.5 cm
    print(f"Readings fitted: {len(res)} | median |residual| {np.median(np.abs(res)):.2f} cm | "
          f"within 0.5 cm: {100 * np.mean(np.abs(res) < 0.5):.0f}%")
    if cal["beta"] is None:
        # coeficiente de correlação r-beta: rho = cov(r, beta) / (sigma_r sigma_beta).
        # |rho| perto de 1 = degenerescência raio-beta (os dados não os separam)
        rho = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2]) if cov[0, 0] * cov[2, 2] > 0 else 0
        print(f"Beam half-angle beta = {p[2]:.1f} ± {sp[2]:.1f} deg (fitted; correlation with r: {rho:+.2f})")
        if abs(rho) > 0.7:
            print("  WARNING: radius and beam width trade off in this fit. Measure beta with the rod")
            print("  sweep and set CAL['beta'] = (value, sigma) for a trustworthy diameter.")
    else:
        print(f"Beam half-angle beta = {p[2]:.1f} ± {CAL['beta'][1]:.1f} deg (fixed, from the rod sweep)")
    # quanto o objeto estava descentrado do eixo do prato em cada varrimento
    for k in range(len(scans)):
        print(f"Centre offset, {scans[k]['name']}: {np.hypot(p[3 + 2 * k], p[4 + 2 * k]):.2f} cm")

    # orçamento de incerteza: uma linha por parte, e a calibração discriminada por fonte
    print("\n=== Uncertainty budget (cm) ===")
    print(f"{'part':14s}{'diameter':>10s}{'height':>10s}")
    for part in sig["D"]:
        print(f"{part:14s}{sig['D'][part]:10.2f}{sig['h'][part]:10.2f}")
    if contrib:
        print("  calibration, by source:")
        for name, (cD, ch) in contrib.items():
            print(f"    {name:12s}{cD:10.2f}{ch:10.2f}")
    print(f"{'TOTAL':14s}{total['D']:10.2f}{total['h']:10.2f}")

    print("\n=== Result ===")
    print(f"Diameter D = {best['D']:.2f} ± {total['D']:.2f} cm   (raw point cloud: {D_raw:.2f} cm)")
    if h_ok:
        # propagação para V = pi D^2 h / 4: (sV/V)^2 = (2 sD/D)^2 + (sh/h)^2
        # (D e h tratados como independentes)
        sV = best["V"] * np.hypot(2 * total["D"] / best["D"], total["h"] / best["h"])
        print(f"Height   h = {best['h']:.2f} ± {total['h']:.2f} cm")
        print(f"Volume   V = {best['V']:.1f} ± {sV:.1f} cm^3")
    else:
        print(f"Height: NOT constrained by these scans (sigma {total['h']:.1f} cm) - "
              f"add a scan from another height or flatter tilts")
    if per:
        print("Per scan:", ", ".join(f"{s['name']}: D={q['D']:.2f} h={q['h']:.2f}"
                                    for s, q in zip(scans, per)))

    # comparação com o paquímetro: z-score com as duas incertezas em quadratura
    if args.D:
        Dr, sDr = args.D
        z = abs(best["D"] - Dr) / np.hypot(total["D"], sDr)
        # o mesmo para a nuvem bruta (com a mesma sigma), para mostrar o enviesamento sem modelo
        zr = abs(D_raw - Dr) / np.hypot(total["D"], sDr)
        print(f"\nvs caliper D = {Dr:.2f} ± {sDr:.2f} cm: difference {best['D'] - Dr:+.2f} cm, z = {z:.1f}"
              f"   (raw cloud: {D_raw - Dr:+.2f} cm, z = {zr:.1f})")
    if args.h and h_ok:
        hr, shr = args.h
        z = abs(best["h"] - hr) / np.hypot(total["h"], shr)
        print(f"vs caliper h = {hr:.2f} ± {shr:.2f} cm: difference {best['h'] - hr:+.2f} cm, z = {z:.1f}")
    print("z < 2 consistent, 2-3 tension, > 3 find the cause")

    # ---------------- figura ----------------
    # à esquerda o perfil lateral (distância ao eixo do objeto vs altura), à direita os
    # resíduos vs ângulo do prato
    base = os.path.splitext(args.logs[0])[0] + "_dimensoes"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    off = 0   # posição do primeiro resíduo de cada varrimento no vetor concatenado res
    for k, (s, g) in enumerate(zip(scans, geos)):
        sx, sz, ux, uz, dist = g
        # pontos brutos no eixo do feixe (plano do lab.)
        x = sx + dist * ux
        z = sz + dist * uz
        # passados para o referencial do prato (mesma convenção de sinais de predict) e
        # centrados no eixo do cilindro ajustado
        ang = np.radians(s["d"][:, 1])
        xo = x * np.cos(ang) - p[3 + 2 * k]
        yo = -x * np.sin(ang) - p[4 + 2 * k]
        a1.scatter(np.hypot(xo, yo), z, s=8, alpha=0.6, label=f"{s['name']} (H={s['H']})")
        n = len(s["d"])
        a2.scatter(s["d"][:, 1], res[off:off + n], s=8, alpha=0.6)
        off += n
        a1.plot(s["R"], s["H"], "ks", ms=7)   # posição do sensor (quadrado preto)
    D, h = best["D"], best["h"]
    # perfil ajustado (parede em r = D/2 e topo em h) e faixa de +-1 sigma do raio
    a1.plot([D / 2, D / 2, 0], [0, h, h], "r-", lw=2, label=f"fit: D = {D:.2f} ± {total['D']:.2f} cm")
    a1.axvspan(D / 2 - total["D"] / 2, D / 2 + total["D"] / 2, color="r", alpha=0.12)
    if args.D:
        # perfil do paquímetro (tracejado); sem altura do paquímetro usa-se a ajustada
        hr = args.h[0] if args.h else h
        a1.plot([args.D[0] / 2, args.D[0] / 2, 0], [0, hr, hr], "k--", lw=1.5,
                label=f"caliper: D = {args.D[0]:.2f} cm")
    a1.set_xlabel("distance from the object axis (cm)")
    a1.set_ylabel("height z (cm)")
    a1.set_title("Side profile: points (raw, on the beam axis) and fit")
    a1.set_xlim(0, None); a1.set_ylim(0, None); a1.set_aspect("equal"); a1.legend(fontsize=8)
    # resíduos: devem estar centrados em 0, sem tendência com o ângulo; a faixa cinzenta é
    # +-ROBUST_SCALE (fora dela a perda robusta reduz o peso da leitura)
    a2.axhline(0, color="k", lw=1)
    a2.axhspan(-ROBUST_SCALE, ROBUST_SCALE, color="gray", alpha=0.2)
    a2.set_xlabel("plate angle (deg)")
    a2.set_ylabel("measured - model (cm)")
    a2.set_title("Residuals: no trend expected")
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=130)
    print(f"\nFigure: {base}.png")


if __name__ == "__main__":
    main()
