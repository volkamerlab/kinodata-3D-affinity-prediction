import sys

from bs4 import BeautifulSoup
import bs4


def main(ident):
    with open(f"data/interaction_analysis/{ident}/report.xml") as f:
        inp = f.read()
    bs = BeautifulSoup(inp, features="xml")

    site = [
        s
        for s in bs.find_all("bindingsite")
        if "UNL" == next(s.find("interactions").find("restype_lig").children)
    ]

    assert len(site) == 1
    site = site[0]

    interactions = site.find("interactions")
    with open(f"data/interaction_analysis/{ident}/report.csv", "w") as f:
        f.write("interaction_type,residue\n")
        for cat_interactions in interactions.contents:
            if not isinstance(cat_interactions, bs4.element.Tag):
                continue
            for r in set([int(r.string) for r in cat_interactions.find_all("resnr")]):
                f.write(f"{cat_interactions.name},{r}\n")


if __name__ == "__main__":
    ident = int(sys.argv[1])
    main(ident)
