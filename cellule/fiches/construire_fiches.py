# -*- coding: utf-8 -*-
"""
Regenere les blocs mesures des fiches 2 et 3.

    python fiches/construire_fiches.py

Trois blocs de ces fiches ne sont pas ecrits a la main : ils sont
*mesures* sur le simulateur, puis projetes en SVG.

  fiche 2, section 04   les trois chemins -- movj, movl, movc -- entre le
                        meme depart et la meme arrivee, avec leurs durees
  fiche 2, section 07   le coin agrandi au lissage 0, 50 et 100 %, et le
                        tableau coin coupe / duree / temps gagne
  fiche 3, conventions  l'ecriture d'une matrice a la main

Ces blocs se periment des que les lois horaires, les profils ou le
lissage changent : relancer le script les remet d'accord avec le code.
Il reimprime les valeurs mesurees pour qu'on puisse les confronter a ce
qu'affichent les fiches.

Les insertions sont encadrees par des marqueurs
<!-- memo:nom --> ... <!-- /memo:nom --> : relancer le script remplace le
bloc au lieu de l'empiler. Le reste des fiches n'est jamais touche, y
compris fiche_index.html, qui s'ecrit a la main comme les autres.
"""
from __future__ import annotations

import io
import os
import re
import sys

import numpy as np

FICHES = os.path.dirname(os.path.abspath(__file__))
PROJET = os.path.dirname(FICHES)
sys.path.insert(0, PROJET)

from dobot_sim import Pilote  # noqa: E402


def trans(x=0.0, y=0.0, z=0.0):
    return np.array([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


RX180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                  [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)


def chemin(construire, **kw):
    """Joue un programme et renvoie (chemin du TCP, duree) a partir du depart."""
    r = Pilote(nom="figure", outil="ventouse", **kw)
    depart = construire(r)
    e = r.trajectoire().echantillonner(0.002)
    P, t = e["P"], e["t"]
    i = int(np.argmin(np.linalg.norm(P - depart[:3, 3], axis=1)))
    return P[i:], float(t[-1] - t[i])


def polyligne(points, projeter, pas=2, seuil=2.0):
    """Points SVG, allegee : on ne garde que ce qui se voit."""
    pts = [projeter(p) for p in points[::pas]]
    gardes = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - gardes[-1][0]) + abs(q[1] - gardes[-1][1]) > seuil:
            gardes.append(q)
    gardes.append(projeter(points[-1]))
    return " ".join("%.1f,%.1f" % q for q in gardes)


def inserer(chemin_fichier, nom, bloc, ancre):
    """Pose ou remplace un bloc balise, juste apres `ancre`. Idempotent."""
    s = io.open(chemin_fichier, encoding="utf-8").read()
    ouvre, ferme = "<!-- memo:%s -->" % nom, "<!-- /memo:%s -->" % nom
    neuf = "%s\n%s\n  %s" % (ouvre, bloc, ferme)
    if ouvre in s:
        s = re.sub(re.escape(ouvre) + r".*?" + re.escape(ferme), lambda m: neuf,
                   s, flags=re.S)
        action = "remplace"
    else:
        if ancre not in s:
            raise SystemExit("ancre introuvable dans %s : %r"
                             % (chemin_fichier, ancre[:70]))
        s = s.replace(ancre, ancre + "\n\n  " + neuf, 1)
        action = "insere"
    io.open(chemin_fichier, "w", encoding="utf-8", newline="\n").write(s)
    print("  %-9s %-20s dans %s" % (action, nom, os.path.basename(chemin_fichier)))


def virgule(x, n=2):
    return ("%.*f" % (n, x)).replace(".", ",")


# ===========================================================================
# Figure 1 : les trois mouvements, meme depart et meme arrivee, vue de dessus
# ===========================================================================
A = trans(200, -110, 40) @ RX180
B = trans(200, 110, 40) @ RX180
VIA = trans(285, 0, 40) @ RX180

mouv = {}
for nom, cmd in (("movj", lambda r: r.movj(B)),
                 ("movl", lambda r: r.movl(B)),
                 ("movc", lambda r: r.movc(via=VIA, T=B))):
    def construire(r, cmd=cmd):
        r.movl(A)
        cmd(r)
        return A
    mouv[nom] = chemin(construire, vitesse=150, acceleration=600,
                       vitesse_articulaire=60)

W1, H1, M1 = 560, 300, 36
xs = np.concatenate([c[:, 0] for c, _ in mouv.values()])
ys = np.concatenate([c[:, 1] for c, _ in mouv.values()])
s1 = min((W1 - 2 * M1) / (ys.max() - ys.min()),
         (H1 - 2 * M1) / (xs.max() - xs.min()))


def p1(p):                      # vue de dessus : y a droite, x vers le haut
    return (M1 + (p[1] - ys.min()) * s1, H1 - M1 - (p[0] - xs.min()) * s1)


COULEURS = {"movj": "var(--axe-y)", "movl": "var(--accent)", "movc": "var(--axe-x)"}
traces = ['      <polyline points="%s" fill="none" stroke="%s" stroke-width="2.2" '
          'stroke-linejoin="round"/>' % (polyligne(c, p1), COULEURS[nom])
          for nom, (c, _) in mouv.items()]

ax, ay = p1(A[:3, 3])
bx, by = p1(B[:3, 3])
vx, vy = p1(VIA[:3, 3])
jx, jy = p1(mouv["movj"][0][len(mouv["movj"][0]) // 2])
cx, cy = p1(mouv["movc"][0][len(mouv["movc"][0]) // 2])
dj, dl, dc = mouv["movj"][1], mouv["movl"][1], mouv["movc"][1]
ecart_j = np.abs(mouv["movj"][0][:, 0]
                 - np.interp(mouv["movj"][0][:, 1], [A[1, 3], B[1, 3]],
                             [A[0, 3], B[0, 3]])).max()

fig_mouvements = """  <div class="schema">
    <svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-mouv">
      <title id="t-mouv">Vue de dessus : les chemins suivis par la pointe pour
      movj, movl et movc entre les deux memes points</title>
{traces}
      <circle class="s-pt" cx="{ax:.0f}" cy="{ay:.0f}" r="4.5"/>
      <circle class="s-pt" cx="{bx:.0f}" cy="{by:.0f}" r="4.5"/>
      <circle cx="{vx:.0f}" cy="{vy:.0f}" r="3.5" fill="var(--axe-x)"/>
      <text class="s-cle" x="{ax:.0f}" y="{ayt:.0f}" text-anchor="middle">depart</text>
      <text class="s-cle" x="{bx:.0f}" y="{byt:.0f}" text-anchor="middle">arrivee</text>
      <text class="s-mut" x="{vx:.0f}" y="{vyt:.0f}" text-anchor="middle">point de passage</text>
      <text class="s-cle" x="{jx:.0f}" y="{jyt:.0f}" text-anchor="middle" fill="var(--axe-y)">movj() &#8212; {dj} s</text>
      <text class="s-cle" x="{mx:.0f}" y="{myt:.0f}" text-anchor="middle" fill="var(--accent)">movl() &#8212; {dl} s</text>
      <text class="s-cle" x="{cx:.0f}" y="{cyt:.0f}" text-anchor="middle" fill="var(--axe-x)">movc() &#8212; {dc} s</text>
    </svg>
  </div>
  <p class="legende-mat">Les trois chemins r&eacute;ellement calcul&eacute;s entre le m&ecirc;me
  d&eacute;part et la m&ecirc;me arriv&eacute;e, vus de dessus, &agrave; 150&nbsp;mm/s et
  60&nbsp;deg/s. <strong>Le plus rapide est celui qui ne tient pas le
  chemin</strong>&nbsp;: {ej}&nbsp;mm d'&eacute;cart &agrave; la droite pour
  <code>movj()</code>, mais {dj}&nbsp;s contre {dl}&nbsp;s.</p>""".format(
    W=W1, H=H1, traces="\n".join(traces),
    ax=ax, ay=ay, ayt=ay + 22, bx=bx, by=by, byt=by + 22,
    vx=vx, vy=vy, vyt=vy - 12, jx=jx, jyt=jy - 12,
    mx=(ax + bx) / 2, myt=by - 12, cx=cx, cyt=cy + 22,
    dj=virgule(dj), dl=virgule(dl), dc=virgule(dc), ej=virgule(ecart_j, 1))

# ===========================================================================
# Figure 2 : le lissage, un coin a angle droit vu de dessus et agrandi
# ===========================================================================
CA = trans(185, -70, 60) @ RX180
CC = trans(265, -70, 60) @ RX180        # le coin
CB = trans(265, 70, 60) @ RX180

liss = {}
for L in (0, 25, 50, 75, 100):
    def construire(r, L=L):
        r.movl(CA)
        r.movl(CC, lissage=L)
        r.movl(CB, lissage=L)
        return CA
    c, duree = chemin(construire, vitesse=150, acceleration=600)
    liss[L] = (c, duree, float(np.min(np.linalg.norm(c - CC[:3, 3], axis=1))))

X0, X1 = CC[0, 3] - 26, CC[0, 3] + 4
Y0, Y1 = CC[1, 3] - 5, CC[1, 3] + 51
W2, H2, M2 = 520, 300, 30
s2 = min((W2 - 2 * M2) / (Y1 - Y0), (H2 - 2 * M2) / (X1 - X0))


def p2(p):
    return (M2 + (p[1] - Y0) * s2, H2 - M2 - (p[0] - X0) * s2)


STYLES = ((0, "var(--encre2)", "2.4"), (50, "var(--accent)", "2.2"),
          (100, "var(--axe-x)", "2.2"))
traces2 = []
for L, col, ep in STYLES:
    c = liss[L][0]
    m = ((c[:, 0] >= X0) & (c[:, 0] <= X1) & (c[:, 1] >= Y0) & (c[:, 1] <= Y1))
    traces2.append('      <polyline points="%s" fill="none" stroke="%s" '
                   'stroke-width="%s" stroke-linejoin="round"/>'
                   % (polyligne(c[m], p2, pas=3, seuil=1.5), col, ep))

kx, ky = p2(CC[:3, 3])
ech = 10 * s2

fig_lissage = """  <div class="schema">
    <svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-liss">
      <title id="t-liss">Agrandissement d'un coin a angle droit : le chemin
      reellement suivi a l'arret au point, puis avec un lissage de 50 et de 100 pour cent</title>
{traces}
      <circle class="s-pt" cx="{kx:.0f}" cy="{ky:.0f}" r="4.5"/>
      <text class="s-mut" x="{kxt:.0f}" y="{kyt:.0f}">le point programme</text>
      <text class="s-cle" x="{lx:.0f}" y="{l0:.0f}" fill="var(--encre2)">arr&#234;t au point &#8212; coin atteint</text>
      <text class="s-cle" x="{lx:.0f}" y="{l50:.0f}" fill="var(--accent)">lissage 50 &#8212; {d50} mm</text>
      <text class="s-cle" x="{lx:.0f}" y="{l100:.0f}" fill="var(--axe-x)">lissage 100 &#8212; {d100} mm</text>
      <line class="s-cote" x1="{bx:.0f}" y1="{by:.0f}" x2="{bx2:.0f}" y2="{by:.0f}"/>
      <line class="s-cote" x1="{bx:.0f}" y1="{by1:.0f}" x2="{bx:.0f}" y2="{by2:.0f}"/>
      <line class="s-cote" x1="{bx2:.0f}" y1="{by1:.0f}" x2="{bx2:.0f}" y2="{by2:.0f}"/>
      <text class="s-mut" x="{bxm:.0f}" y="{byt:.0f}" text-anchor="middle">10 mm</text>
    </svg>
  </div>""".format(
    W=W2, H=H2, traces="\n".join(traces2),
    kx=kx, ky=ky, kxt=kx + 20, kyt=ky + 32,
    lx=W2 - 190, l0=H2 - 88, l50=H2 - 66, l100=H2 - 44,
    bx=W2 - 190, by=H2 - 22, bx2=W2 - 190 + ech,
    by1=H2 - 27, by2=H2 - 17, bxm=W2 - 190 + ech / 2, byt=H2 - 32,
    d50=virgule(liss[50][2]), d100=virgule(liss[100][2]))

# ===========================================================================
# Mesure : une action qui suit un mouvement impose-t-elle l'arret au point ?
# ===========================================================================
HAUT = trans(220, -60, 110) @ RX180
PRI = trans(220, -60, 25) @ RX180
COTE = trans(150, -60, 110) @ RX180

action = {}
for L in (0, 80):
    r = Pilote(nom="action", outil="ventouse", vitesse=150, acceleration=600)
    r.movl(HAUT)
    r.movl(PRI, lissage=L)
    r.ventouse(True)
    r.movl(HAUT, lissage=L)
    r.movl(COTE, lissage=L)
    e = r.trajectoire().echantillonner(0.001)
    ev = [x for x in r.programme.evenements if x.type == "OUTIL"][0]
    i = int(np.searchsorted(e["t"], ev.t))
    action[L] = (float(np.linalg.norm(e["P"][i] - PRI[:3, 3])), float(e["V"][i]))

sans_action = {}
for L in (0, 80):
    r = Pilote(nom="sans", outil="ventouse", vitesse=150, acceleration=600)
    r.movl(HAUT)
    r.movl(PRI, lissage=L)
    r.movl(COTE, lissage=L)
    e = r.trajectoire().echantillonner(0.001)
    sans_action[L] = float(np.min(np.linalg.norm(e["P"] - PRI[:3, 3], axis=1)))


lignes = "\n".join(
    "    <tr><td>%s</td><td>%s mm</td><td>%s s</td><td>%s</td></tr>"
    % ("<strong>arr&ecirc;t au point</strong>" if L == 0 else "lissage %d %%" % L,
       virgule(liss[L][2]), virgule(liss[L][1]),
       "&mdash;" if L == 0
       else "%d %%" % round(100 * (liss[0][1] - liss[L][1]) / liss[0][1]))
    for L in (0, 25, 50, 75, 100))

bloc_lissage = """  <h3>Deux choses distinctes</h3>
  <p>Le param&egrave;tre est le m&ecirc;me, mais les deux comportements n'ont ni le
  m&ecirc;me usage ni le m&ecirc;me nom.</p>
  <div class="cadre">
  <table>
    <tr><th>On &eacute;crit</th><th>Comportement</th><th>Le point est&hellip;</th><th>Pour</th></tr>
    <tr><td><code>robot.movl(PRISE)</code></td>
        <td><strong>arr&ecirc;t au point</strong></td>
        <td>atteint, vitesse nulle</td>
        <td>prise, d&eacute;pose, toute position de travail</td></tr>
    <tr><td><code>robot.movl(HAUT, lissage=60)</code></td>
        <td><strong>lissage de 60&nbsp;%</strong></td>
        <td>manqu&eacute;, le coin est coup&eacute;</td>
        <td>d&eacute;gagements et transferts en l'air</td></tr>
  </table>
  </div>
  <p>L'arr&ecirc;t au point ne se demande pas&nbsp;: c'est ce qu'on obtient en ne
  donnant <strong>aucune</strong> valeur de lissage. Le lissage, lui, s'&eacute;crit
  toujours explicitement, avec son pourcentage.</p>

  <h3>Pourquoi la distinction compte&nbsp;: les actions</h3>
  <p>D&egrave;s qu'une <strong>action</strong> suit un mouvement &mdash; ventouse,
  pince, sortie tout-ou-rien &mdash; il faut que le mouvement soit
  <strong>termin&eacute;</strong> avant qu'elle parte. Aspirer au-dessus de la
  pi&egrave;ce, en marche, ne prend rien.</p>
  <p>Le mod&egrave;le ne vous prot&egrave;ge pas&nbsp;: une action ne demande aucun temps,
  elle part &agrave; l'instant o&ugrave; la file l'atteint. Si le mouvement qui la
  pr&eacute;c&egrave;de est liss&eacute;, cet instant arrive <strong>avant</strong> la cible.
  Mesur&eacute; &agrave; l'instant pr&eacute;cis o&ugrave; la ventouse s'enclenche&nbsp;:</p>

  <div class="cadre">
  <table>
    <tr><th>Le mouvement qui pr&eacute;c&egrave;de l'action</th><th>Distance &agrave; la cible</th><th>Vitesse</th></tr>
    <tr><td><strong>arr&ecirc;t au point</strong></td>
        <td><strong>{a0d} mm</strong></td><td><strong>{a0v} mm/s</strong> &mdash; pos&eacute;e sur la pi&egrave;ce</td></tr>
    <tr><td><code>lissage=80</code></td>
        <td><strong>{a8d} mm</strong></td><td><strong>{a8v} mm/s</strong> &mdash; encore en l'air, en marche</td></tr>
  </table>
  </div>

  <p>Un demi-millim&egrave;tre et seize millim&egrave;tres par seconde&nbsp;: la ventouse
  aspire avant d'&ecirc;tre pos&eacute;e. Rien dans le mod&egrave;le ne le corrige, et c'est
  d&eacute;lib&eacute;r&eacute; &mdash; <strong>d&eacute;cider o&ugrave; le robot doit s'arr&ecirc;ter fait partie
  du travail de programmation</strong>, ce n'est pas au simulateur de le
  deviner.</p>
  <p>D'o&ugrave; la r&egrave;gle, qui est une r&egrave;gle d'atelier et pas une
  particularit&eacute; du logiciel&nbsp;: <strong>on ne lisse pas un point o&ugrave; l'on
  agit</strong>. Un point de travail se programme &agrave; l'arr&ecirc;t au point, un
  d&eacute;gagement avec un lissage.</p>
  <p>Une <strong>temporisation</strong>, elle, n'est jamais absorb&eacute;e&nbsp;:
  <code>attendre(0.3)</code> et l'attente d'une entr&eacute;e sont des temps
  demand&eacute;s, le lissage ne peut pas les rogner.</p>

<pre><code>robot.movl(PRISE)                      <span class="c"># ARRET AU POINT : on prend la piece</span>
robot.ventouse(<span class="k">True</span>)                 <span class="c"># part a l'arret, pointe sur la piece</span>
robot.attendre(<span class="n">0.3</span>)
robot.movl(PRISE @ HAUT, lissage=<span class="n">60</span>)   <span class="c"># LISSAGE : degagement, le coin peut etre coupe</span>
robot.movj(DEPOSE @ HAUT, lissage=<span class="n">60</span>)  <span class="c"># LISSAGE : transfert en l'air</span>
robot.movl(DEPOSE)                     <span class="c"># ARRET AU POINT : on depose</span></code></pre>

  <h3>Ce que co&ucirc;te le lissage</h3>
{figure}
  <p>Un coin &agrave; angle droit, 150&nbsp;mm/s, vu de dessus et agrandi. Mesur&eacute;
  sur la trajectoire &eacute;chantillonn&eacute;e&nbsp;:</p>

  <div class="cadre">
  <table>
    <tr><th>R&eacute;glage</th><th>Coin coup&eacute;</th><th>Dur&eacute;e</th><th>Temps gagn&eacute;</th></tr>
{lignes}
  </table>
  </div>

  <p>Le march&eacute; est explicite&nbsp;: <strong>{gain}&nbsp;% de temps contre
  {cout}&nbsp;mm de pr&eacute;cision</strong>. La section <em>contr&ocirc;le du
  mouvement</em> de <code>mon_programme.py</code> donne, instruction par
  instruction, la distance minimale de la pointe &agrave; sa cible&nbsp;: un point
  atteint y est &agrave; 0,00&nbsp;mm, un point liss&eacute; est volontairement
  manqu&eacute;.</p>""".format(
    figure=fig_lissage, lignes=lignes,
    gain=round(100 * (liss[0][1] - liss[100][1]) / liss[0][1]),
    cout=virgule(liss[100][2]),
    a0d=virgule(action[0][0], 3), a0v=virgule(action[0][1], 2),
    a8d=virgule(action[80][0], 3), a8v=virgule(action[80][1], 2))

# ===========================================================================
# Bloc pour la fiche 3 : ecrire une matrice a la main
# ===========================================================================
bloc_matrice = """  <h3>&Eacute;crire une matrice &agrave; la main</h3>
  <p>Les fabriques <code>cible()</code> et <code>repere_travail()</code> ne font
  qu'assembler des nombres dans un tableau&nbsp;: rien n'emp&ecirc;che de l'&eacute;crire
  directement, et c'est souvent plus clair pour poser une convention.</p>
<pre><code><span class="c"># un REPERE DE TRAVAIL : axe z vers le HAUT, rotation = identite</span>
MAGASIN = np.array([[<span class="n">1</span>, <span class="n">0</span>, <span class="n">0</span>, <span class="n">215</span>],
                    [<span class="n">0</span>, <span class="n">1</span>, <span class="n">0</span>, <span class="n">-95</span>],
                    [<span class="n">0</span>, <span class="n">0</span>, <span class="n">1</span>,  <span class="n">25</span>],
                    [<span class="n">0</span>, <span class="n">0</span>, <span class="n">0</span>,   <span class="n">1</span>]], dtype=float)

<span class="c"># une CIBLE au meme endroit : axe z retourne vers le BAS</span>
PRISE = np.array([[<span class="n">1</span>,  <span class="n">0</span>,  <span class="n">0</span>, <span class="n">215</span>],
                  [<span class="n">0</span>, <span class="n">-1</span>,  <span class="n">0</span>, <span class="n">-95</span>],
                  [<span class="n">0</span>,  <span class="n">0</span>, <span class="n">-1</span>,  <span class="n">25</span>],
                  [<span class="n">0</span>,  <span class="n">0</span>,  <span class="n">0</span>,   <span class="n">1</span>]], dtype=float)

<span class="c"># un REPERE OUTIL : de l'axe de rotation a la pointe, z vers le BAS</span>
OUTIL = np.array([[<span class="n">1</span>,  <span class="n">0</span>,  <span class="n">0</span>,    <span class="n">0</span>],
                  [<span class="n">0</span>, <span class="n">-1</span>,  <span class="n">0</span>,    <span class="n">0</span>],
                  [<span class="n">0</span>,  <span class="n">0</span>, <span class="n">-1</span>, <span class="n">-79.3</span>],
                  [<span class="n">0</span>,  <span class="n">0</span>,  <span class="n">0</span>,    <span class="n">1</span>]], dtype=float)</code></pre>
  <p>Deux matrices suffisent ensuite &agrave; tout composer&nbsp;: le retournement
  <code>Rx(180&deg;)</code>, qui fait d'un rep&egrave;re une cible, et une translation
  pure. L'op&eacute;rateur de composition est <code>@</code>&nbsp;; post-multiplier,
  c'est se d&eacute;placer dans le rep&egrave;re de l'<strong>outil</strong>,
  pr&eacute;-multiplier dans celui de la <strong>base</strong>.</p>"""

# ===========================================================================
# Versement dans les fiches
# ===========================================================================
print("\nversements :")
inserer(os.path.join(FICHES, "fiche_programmation.html"), "figure-mouvements",
        fig_mouvements,
        "<strong>fa&ccedil;on d'aller</strong> &agrave; la cible change &mdash; et les "
        "unit&eacute;s qui vont avec.</p>")
inserer(os.path.join(FICHES, "fiche_programmation.html"), "lissage-mesure",
        bloc_lissage,
        "<p><code>robot.resume()</code> annonce le temps gagn&eacute; par le lissage "
        "sur l'ensemble\n  du programme&nbsp;: c'est la mesure qui justifie le "
        "r&eacute;glage.</p>")
inserer(os.path.join(FICHES, "fiche_api.html"), "matrice-a-la-main",
        bloc_matrice,
        "possible seulement parce que la m&eacute;canique du Magician impose "
        "l'orientation de\n  l'outil.</p>")

print("\nfigure 1 -- les trois mouvements (meme depart, meme arrivee)")
for nom, (c, duree) in mouv.items():
    droite = np.interp(c[:, 1], [A[1, 3], B[1, 3]], [A[0, 3], B[0, 3]])
    print("   %-7s ecart max a la droite %5.1f mm   duree %.2f s"
          % (nom + "()", np.abs(c[:, 0] - droite).max(), duree))
print("\nfigure 2 -- le lissage (coin a angle droit, 150 mm/s)")
for L in (0, 25, 50, 75, 100):
    print("   lissage %3d %%   coin coupe %5.2f mm   duree %.2f s"
          % (L, liss[L][2], liss[L][1]))
