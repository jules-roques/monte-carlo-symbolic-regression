from mcsr.utils.dataloader import SRSDLoader

loader = SRSDLoader(difficulty="easy", splits=["train", "validation"])


print(loader[0].keys())
