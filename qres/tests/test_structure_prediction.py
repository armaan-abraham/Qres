import time
from qres.structure_prediction import StructurePredictor
import torch


structure_predictor = StructurePredictor()

test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

with open("test_protein.txt", "w") as f:
    # write the file
    result = structure_predictor.predict_structure([test_protein[:50]])[0]
    f.write(result)

exit()
tpp = []
for i in range(15):
    test_proteins = [test_protein[:50] for j in range(i + 1)]
    start = time.time()
    infer_structure_batch(test_proteins)
    tpp.append((time.time() - start) / (i + 1))
ax = sns.lineplot(x=range(len(tpp)), y=tpp)
# print to svg
plt.savefig("test.svg")
