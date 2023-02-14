This repo was used to train Lepard. The weights resulting from Lepard were then used in NDP. To train you can write:

```
python main.py configs/train/_astrivis_fcgf.py
python main.py configs/train/_astrivis.py
```

The first is for training with the FCGF feature extractor and the second with the KPFCN feature extractor.
