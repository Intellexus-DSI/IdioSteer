import csv, os, idiosteer
D=idiosteer.DATA
def test_idiosteer_prefixes():
    rows=list(csv.DictReader(open(os.path.join(D,"idiosteer","idiosteer_prefixes.csv"))))
    assert len(rows)==100
    assert len({r["idiom"] for r in rows})==20
def test_concreteness_lists_present():
    for f in ["concrete_250.txt","abstract_250.txt","gj_concrete_500.txt","gj_abstract_500.txt"]:
        assert os.path.getsize(os.path.join(D,"concreteness",f))>0
def test_idiolink_subset():
    rows=list(csv.reader(open(os.path.join(D,"idiolink_subset","retained_ids.csv"))))
    assert len(rows)>1000
