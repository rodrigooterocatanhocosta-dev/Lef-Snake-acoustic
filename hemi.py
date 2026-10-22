"""hemi.py - modelo de primeiro eco para uma MEIA esfera (face plana para cima ou para baixo),
por traçado de raios dentro do cone do feixe.

O QUE FAZ
  Objeto: bola de raio a e centro (cx, cy, cz), cortada pelo plano horizontal z = cz.
    'down' = taça (fica a metade de baixo; face plana em cima, a z = cz);
    'up'   = cúpula (fica a metade de cima; face plana do lado do prato, a z = cz).
  Para cada leitura prevê a distância do primeiro eco e ajusta (cx, cy, cz, a) às distâncias medidas.
  É usado como módulo (por globo_resultados.py); não tem linha de comandos própria.
  Exemplo:  import hemi;  p, e, sig, cost, n = hemi.fit("globo12.txt", "up")

PORQUÊ TRAÇADO DE RAIOS
  Para a esfera completa há fórmula fechada (esfera.py). Com a face plana e o prato, o ponto mais
  próximo dentro do cone pode estar na calote, no bordo, na face plana ou no próprio prato, e não há
  fórmula simples. Solução: substitui-se o cone por um feixe de ~97 raios (o eixo + 6 anéis x 16
  azimutes), calcula-se para cada raio a distância à primeira superfície atingida e fica-se com a
  menor de todas - é essa a distância que o HC-SR04 reporta (o primeiro eco que regressa).

MÉTODO, PASSO A PASSO
  1. cone_dirs(): direções dos raios dentro do cone de meia-abertura beta, à volta de cada eixo U.
  2. first_echo(): interseção de cada raio com (i) a superfície esférica, só na metade que existe,
     (ii) o disco da face plana, (iii) o prato z = 0; distância mínima sobre todos os raios.
  3. fit(): mínimos quadrados robustos (soft_l1); parâmetros podem ser fixados (valor ou função);
     sigma de uma leitura pela MAD; incertezas dos parâmetros pela covariância.

SAÍDAS
  fit() devolve parâmetros, incertezas, sigma dos resíduos, custo final e número de leituras.

LIMITES CONHECIDOS
  - O cone é amostrado (97 raios), por isso a previsão é ligeiramente "aos degraus" quando os
    parâmetros variam; daí o diff_step largo no ajuste.
  - Feixe tratado como cone de bordo nítido, sem perfil de intensidade nem ecos múltiplos.
  - O prato é um plano infinito em z = 0.
"""
import numpy as np
from scipy.optimize import least_squares
import esfera as E

def cone_dirs(U, beta, n_ring=6, n_az=16):
    """Direções dos raios que amostram o cone do feixe.

    Entradas:
       U      - (N, 3) eixos do feixe (vetores unitários), um por leitura;
       beta   - meia-abertura do cone (graus);
       n_ring - número de anéis (ângulos beta/6, 2beta/6, ..., beta);
       n_az   - número de raios por anel (azimutes igualmente espaçados).
    Saída: (N, M, 3) vetores unitários, M = 1 + n_ring*n_az = 97 por defeito (eixo incluído).
    A amostragem é uniforme em ângulo, não em ângulo sólido: não importa, porque o primeiro eco é um
    MÍNIMO sobre os raios (não uma média); o que importa é cobrir o cone até ao bordo.
    """
    # Referencial local para cada eixo do feixe: a = U x z (horizontal, perpendicular ao feixe),
    # b = U x a (completa a base ortonormada). Funciona porque o feixe nunca é vertical.
    a = np.cross(U, [0, 0, 1.0]); a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(U, a)
    dirs = [U]
    # Raio com ângulo th ao eixo e azimute ph: cos(th) U + sin(th) (cos(ph) a + sin(ph) b).
    for k in range(1, n_ring + 1):
        th = np.radians(beta) * k / n_ring
        for j in range(n_az):
            ph = 2 * np.pi * j / n_az
            dirs.append(np.cos(th) * U + np.sin(th) * (np.cos(ph) * a + np.sin(ph) * b))
    return np.stack(dirs, 1)          # (N, M, 3)

def first_echo(p, S, D, orient):
    """Distância do primeiro eco prevista para cada leitura.

    Entradas:
       p      - (cx, cy, cz, a): centro da bola e raio (cm);
       S      - (N, 3) posições do sensor (cm);
       D      - (N, M, 3) direções dos raios do cone (de cone_dirs());
       orient - 'up' (cúpula) ou 'down' (taça).
    Saída: (N,) distância à superfície mais próxima atingida por algum raio do cone (cm);
           inf se nenhum raio atingir nada (não acontece na prática: os raios descendentes batem no prato).
    Para cada raio X(t) = S + t D (t = distância, D unitário) guarda-se o menor t > 0 entre todas as
    superfícies; no fim, mínimo sobre os M raios.
    """
    cx, cy, cz, a = p
    C = np.array([cx, cy, cz])
    O = S[:, None, :] - C              # (N,1,3) sensor relativamente ao centro
    # Interseção raio-esfera: |O + t D|^2 = a^2  <=>  t^2 + 2 (O.D) t + (|O|^2 - a^2) = 0
    # (D unitário). Com bq = O.D e cq = |O|^2 - a^2: t = -bq -/+ sqrt(bq^2 - cq).
    # disc <= 0: o raio não toca a esfera.
    bq = (O * D).sum(-1); cq = (O**2).sum(-1) - a * a
    disc = bq**2 - cq
    t = np.full(disc.shape, np.inf)
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0))
    # Testam-se as duas raízes (entrada e saída): a de entrada pode cair na metade que NÃO existe
    # (foi cortada), e então a superfície vista é a da saída - o interior da taça, por exemplo.
    for tt in (-bq - sq, -bq + sq):
        # Seleção do semi-espaço: z do ponto de impacto relativamente ao centro.
        # 'down' (taça) só tem a metade z <= cz; 'up' (cúpula) só a metade z >= cz.
        z = O[..., 2] + tt * D[..., 2]
        side = (z <= 0) if orient == "down" else (z >= 0)
        good = ok & (tt > 0) & side
        t = np.where(good & (tt < t), tt, t)
    # Face plana: plano z = cz (z relativo = 0): O_z + t D_z = 0  =>  t = -O_z / D_z.
    # Raios quase horizontais (|D_z| ~ 0) não tocam o plano; troca-se D_z por 1e-9 para não
    # dividir por zero (dá um t enorme, que nunca é o mínimo).
    tz = -O[..., 2] / np.where(np.abs(D[..., 2]) > 1e-9, D[..., 2], 1e-9)
    X = O + tz[..., None] * D
    # Teste do disco: o impacto só conta se estiver dentro do círculo da face, x^2 + y^2 <= a^2
    # (coordenadas relativas ao centro).
    good = (tz > 0) & ((X[..., 0]**2 + X[..., 1]**2) <= a * a)
    t = np.where(good & (tz < t), tz, t)
    # O prato (z = 0, absoluto) também é um objeto para o primeiro eco: um raio do cone que chegue
    # ao prato antes de tocar no objeto dá um eco mais curto (S_z + t D_z = 0).
    tp = -S[:, None, 2] / np.where(np.abs(D[..., 2]) > 1e-9, D[..., 2], 1e-9)
    t = np.where((tp > 0) & (tp < t), tp, t)
    # Primeiro eco = o raio do cone que encontra uma superfície mais cedo.
    return t.min(1)

def fit(path, orient, beta=12.4, fix=None, p0=None, cal=None):
    """Ajusta a meia esfera às leituras de um ficheiro.

    Entradas:
       path   - ficheiro de registo;
       orient - 'up' (cúpula) ou 'down' (taça);
       beta   - meia-abertura do feixe (graus), por defeito 12.4 (calibrado na caixa);
       fix    - dicionário de parâmetros fixos {'cx'|'cy'|'cz'|'a': valor ou função(p)}; uma função
                permite restrições, p.ex. {'cz': lambda p: T - p[3]} impõe topo cz + a = T;
       p0     - estimativa inicial (cx, cy, cz, a); por defeito centro no eixo, cz = 7 (taça) ou 0
                (cúpula assente no prato), a = 7 cm;
       cal    - perturbação de calibração passada a esfera.geometry() (ver esfera.py).
    Saídas: (p, e, sig, cost, n)
       p    - (cx, cy, cz, a) ajustados (cm);
       e    - sigma do ajuste de cada parâmetro (0 para os fixos);
       sig  - sigma robusto (MAD) dos resíduos = incerteza de uma leitura (cm);
       cost - valor final da função de custo (usado no perfil de verosimilhança);
       n    - número de leituras.
    """
    d, R, H = E.readings(path)
    S, U, dist = E.geometry(d, R, H, cal or {})
    D = cone_dirs(U, beta)
    names = ["cx", "cy", "cz", "a"]
    p0 = np.array(p0 if p0 is not None else [0, 0, 7.0 if orient == "down" else 0.0, 7.0])
    # Índices dos parâmetros livres (os que o otimizador varia).
    free = [i for i in range(4) if not (fix and names[i] in fix)]
    def full(q):
        """Reconstrói o vetor completo (cx, cy, cz, a) a partir dos livres q, aplicando 'fix'."""
        p = p0.copy(); p[free] = q
        if fix:
            for k, v in fix.items():
                p[names.index(k)] = v(p) if callable(v) else v
        return p
    def res(q):
        """Resíduos previsto - medido (cm), limitados a +-6 cm."""
        # O clip impede que uma leitura absurda (outro objeto, ou t = inf) domine o ajuste;
        # a perda soft_l1 já atenua os resíduos grandes, o clip trava os extremos.
        r = first_echo(full(q), S, D, orient) - dist
        return np.clip(r, -6, 6)
    # soft_l1: quadrática para |res| < f_scale = 0.3 cm, linear para resíduos grandes (robusta a ecos
    # espúrios). diff_step = 0.01: passo relativo das diferenças finitas do jacobiano; tem de ser
    # largo porque o modelo é um mínimo sobre raios discretos (função aos degraus a escala fina).
    s = least_squares(res, p0[free], loss="soft_l1", f_scale=0.3, diff_step=0.01)
    r = res(s.x)
    # sigma de uma leitura: 1.4826*MAD = desvio padrão se gaussiano, insensível aos ecos espúrios.
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    J = s.jac
    # Covariância: sigma^2 (J^T J)^-1, com pseudo-inversa para o caso de parâmetros quase
    # degenerados (p.ex. cz e a quando só se vê a calote): não falha e devolve sigmas grandes.
    cov = sig**2 * np.linalg.pinv(J.T @ J)
    e = np.zeros(4); e[free] = np.sqrt(np.diag(cov))
    return full(s.x), e, sig, s.cost, len(r)
