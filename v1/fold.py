import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

if not torch.cuda.is_available():
    print("WARNING: GPU not found")

torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

if torch.cuda.is_available():
    model = model.cuda()

model.esm = model.esm.half()
# model.trunk.set_chunk_size(64)

def infer_structure_batch(sequences):
    tokenized_input = tokenizer(sequences, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
    if torch.cuda.is_available():
        tokenized_input = tokenized_input.cuda()
    with torch.no_grad():
        outputs = model(tokenized_input)
    start = time.time()
    outputs = convert_outputs_to_pdb(outputs)
    print("convert outputs to pdb", time.time() - start)
    return outputs

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

if __name__ == "__main__":
    test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

    with open("test_protein.txt", "w") as f:
        # write the file
        result = infer_structure_batch([test_protein[:50]])[0]
        f.write(result)


    exit()
    tpp = []
    for i in range(15):
        test_proteins = [test_protein[:50] for j in range(i+1)]
        start = time.time()
        infer_structure_batch(test_proteins)
        tpp.append((time.time() - start) / (i+1))
    ax = sns.lineplot(x=range(len(tpp)), y=tpp)
    # print to svg
    plt.savefig("test.svg")

    # test_proteins = [test_protein, test_protein, test_protein[:10], test_protein[:20], test_protein[:30], test_protein[:50], test_protein[:100], test_protein[:200]]
    # test_proteins = [test_protein]
    # for i in range(50):
    #     test_proteins = [test_protein[:100] for j in range(i+1)]
    #     start = time.time()
    #     infer_structure_batch(test_proteins)
    #     print(len(test_proteins), time.time() - start)


    # start = time.time()
    # infer_structure_batch(test_proteins)
    # print(time.time() - start)




    # for p in test_proteins:
    #     start = time.time()
    #     infer_structure(p)
    #     print(len(p), time.time() - start)

