import sys
try:
    import requests
except ImportError:
    sys.exit("Activate your wildfire env, or run:  pip install requests")

DOIS = [
    (1, "Abatzoglou", "10.1038/sdata.2017.191"),
    (2, "Aguilera", "10.1029/2019GH000225"),
    (3, "Aldersley", "10.1016/j.scitotenv.2011.05.032"),
    (4, "Bansal", "10.1007/978-3-319-91341-4_2"),
    (5, "Benjamini", "10.1111/j.2517-6161.1995.tb02031.x"),
    (6, "?", "10.1109/TEVC.2004.826069"),
    (8, "Bj", "10.1016/j.ecoinf.2021.101397"),
    (9, "Blanco", "10.1016/j.psep.2015.04.002"),
    (10, "Brookes", "10.3389/fevo.2021.676961"),
    (11, "Byrne", "10.1038/s41586-024-07878-z"),
    (12, "Cohen", "10.4324/9780203771587"),
    (13, "Coker", "10.1038/s44407-025-00014-9"),
    (14, "Cottle", "10.1016/j.atmosenv.2014.03.005"),
    (15, "Ding", "10.1016/j.neucom.2020.04.110"),
    (17, "Hanes", "10.1139/cjfr-2018-0293"),
    (18, "Heyerdahl", "10.1139/x11-160"),
    (19, "Hochreiter", "10.1162/neco.1997.9.8.1735"),
    (20, "Hong", "10.1016/j.envpol.2016.10.056"),
    (21, "Hu", "10.1186/s12874-024-02392-2"),
    (22, "Iban", "10.1016/j.ecoinf.2022.101647"),
    (23, "Jain", "10.1038/s41467-024-51154-7"),
    (24, "Johnson", "10.2307/3237276"),
    (25, "Jordan", "10.1071/WF14070"),
    (26, "Juan", "10.1007/s00477-012-0568-y"),
    (27, "Kennedy", "10.1016/j.foreco.2010.05.037"),
    (28, "Kirchmeier-Young", "10.1029/2018EF001050"),
    (29, "Klenner", "10.1016/j.foreco.2008.02.047"),
    (30, "Macias", "10.1098/rstb.2007.2202"),
    (31, "Marcoux", "10.1016/j.foreco.2014.12.027"),
    (32, "McGill", "10.1080/00031305.1978.10479236"),
    (33, "Metsaranta", "10.1016/j.foreco.2022.120729"),
    (34, "Meyn", "10.1007/s10113-012-0319-0"),
    (35, "Parisien", "10.1038/s43247-023-00977-1"),
    (36, "Parisien", "10.1038/s41467-020-15961-y"),
    (37, "Perrakis", "10.4996/fireecology.1002010"),
    (38, "Robichaud", "10.1016/j.geomorph.2013.04.024"),
    (40, "Sharma", "10.1016/j.micpro.2021.104293"),
    (41, "Shen", "10.1038/s41598-019-48995-4"),
    (42, "Shrivastava", "10.1016/j.measen.2022.100657"),
    (43, "Singh", "10.1038/s41598-021-93651-5"),
    (44, "Sun", "10.1016/j.gr.2022.07.013"),
    (45, "Tymstra", "10.1016/j.pdisas.2019.100045"),
    (46, "Umunnakwe", "10.1049/gtd2.12463"),
    (47, "Wentworth", "10.1016/j.atmosenv.2018.01.013"),
    (48, "Whitman", "10.1038/s41598-019-55036-7"),
    (49, "Wulder", "10.1016/j.rse.2009.03.004"),
    (50, "Yang", "10.1016/j.compenvurbsys.2024.102133"),
    (51, "Zeraatpisheh", "10.1016/j.geodrs.2021.e00440"),
    (52, "Zhang", "10.1016/j.ecolind.2021.107735"),
    (53, "Zhang", "10.1155/2015/931256"),
    (54, "Zhao", "10.1016/j.jag.2025.104358"),
]

dead = []
print("Checking %d DOIs via https://doi.org ...\n" % len(DOIS))
for n, author, doi in DOIS:
    try:
        r = requests.head("https://doi.org/" + doi, allow_redirects=True, timeout=25)
        code = r.status_code; ok = code < 400
    except Exception:
        code = "ERR"; ok = False
    print("[%2d] %s %s  %s  (%s)" % (n, "OK  " if ok else "DEAD", code, doi, author))
    if not ok: dead.append((n, author, doi))
print("\n=== %d checked | %d dead ===" % (len(DOIS), len(dead)))
for n, a, d in dead: print("  DEAD [%d] %s: https://doi.org/%s" % (n, a, d))
