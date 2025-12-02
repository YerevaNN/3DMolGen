titan2) ➜  YNNtitan git:(train_conformers) ✗ python torchtitan/count_tokens_and_samples.py --sample-lines 10000

Using train path from config: /nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/train
Using valid path from config: /nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/valid
======================================================================
DATASET SIZE ESTIMATION TOOL
======================================================================
Randomly sampling 10000 lines to estimate full dataset (seq_len=2048)

Loading tokenizer...
Tokenizer loaded. Vocab size: 128330

======================================================================
TRAIN DATASET: conformers_train
======================================================================
Dataset path: /nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/train
Total .jsonl files: 6
Total lines in dataset: 5,745,689
  Randomly sampling 10000 lines from 5,745,689 total lines across 6 files...

FROM RANDOM SAMPLE OF 10000 LINES:
  Characters in sampled lines: 7,859,861
  Tokens produced: 5,001,070
  Samples this represents: 2,441
  Avg chars per line: 786.0
  Avg tokens per line: 500.1
  Lines per sample: 4.10

EXTRAPOLATED TO FULL DATASET:
  Estimated total characters: 4,516,031,688
  Estimated total tokens: 2,873,459,288
  Estimated total samples (seq_len=2048): 1,403,056

SAMPLE:
--- Sample ---
Encoded tokens (first 200 of 440): [128000, 128256, 3791, 16, 42017, 3100, 8368, 11992, 46, 6758, 34, 31, 39, 9725, 66, 17, 641, 38154, 17, 23015, 10, 9725, 28, 46, 6758, 46, 12, 2526, 77, 17, 1031, 1031, 17, 45, 16, 128257, 128328, 44604, 35239, 12, 17, 13, 20772, 19, 5106, 17, 13, 10828, 24, 11, 16, 13, 11483, 31868, 34, 35239, 12, 17, 13, 24970, 18, 5106, 16, 13, 5926, 24, 11, 15, 13, 18277, 23, 29, 16, 5941, 34, 35239, 12, 15, 13, 26537, 20, 5106, 16, 13, 19838, 19, 5106, 15, 13, 12245, 22, 2284, 58, 34, 35239, 12, 15, 13, 8874, 17, 5106, 17, 13, 25091, 18, 5106, 15, 13, 24646, 18, 2284, 58, 52371, 17, 35239, 15, 13, 21385, 16, 5106, 18, 13, 24824, 17, 5106, 16, 13, 10861, 21, 9414, 5941, 46, 35239, 12, 15, 13, 9423, 16, 5106, 18, 13, 19057, 20, 11, 15, 13, 24487, 16, 29, 6758, 34, 31, 39, 35239, 12, 15, 13, 9800, 16, 5106, 15, 13, 13384, 16, 5106, 16, 13, 23904, 17, 2284, 58, 66, 35239, 16, 13, 15537, 23, 5106, 15, 13, 21602, 16, 5106, 15, 13, 22303, 17, 29, 17, 12729, 35239, 17, 13, 9588, 21, 5106, 15, 13, 18625, 5106]

Decoded text:
<|begin_of_text|>[SMILES]CC1=C(C(N)=O)[C@H](c2ccccc2[N+](=O)[O-])n2ncnc2N1[/SMILES][CONFORMER][C]<-2.7704,-2.1829,1.211>[C]<-2.0473,-1.1929,0.3618>1=[C]<-0.8495,-1.3984,-0.2317>([C]<-0.1242,-2.6813,-0.0853>([NH2]<0.6151,-3.0582,-1.1726>)=[O]<-0.1331,-3.3685,0.9161>)[C@H]<-0.1531,-0.3121,-1.0482>([c]<1.2598,-0.0391,-0.5542>2[c]<2.3206,-0.511,-1.3215>[c]<3.6391,-0.3053,-0.961>[c]<3.9421,0.4178,0.1804>[c]<2.9125,0.9311,0.9396>[c]<1.5846,0.7012,0.5867>2[N+]<0.6102,1.292,1.4743>(=[O]<-0.4629,0.7453,1...

DETAILED TOKEN ANALYSIS:
Original processed text (692 chars): [SMILES]CC1=C(C(N)=O)[C@H](c2ccccc2[N+](=O)[O-])n2ncnc2N1[/SMILES][CONFORMER][C]<-2.7704,-2.1829,1.211>[C]<-2.0473,-1.1929,0.3618>1=[C]<-0.8495,-1.3984,-0.2317>([C]<-0.1242,-2.6813,-0.0853>([NH2]<0.61...

Token breakdown (first 100 tokens):
Token  0: 128000 -> '<|begin_of_text|>'
Token  1: 128256 -> '[SMILES]'
Token  2:   3791 -> 'CC'
Token  3:     16 -> '1'
Token  4:  42017 -> '=C'
Token  5:   3100 -> '(C'
Token  6:   8368 -> '(N'
Token  7:  11992 -> ')='
Token  8:     46 -> 'O'
Token  9:   6758 -> ')['
Token 10:     34 -> 'C'
Token 11:     31 -> '@'
Token 12:     39 -> 'H'
Token 13:   9725 -> ']('
Token 14:     66 -> 'c'
Token 15:     17 -> '2'
Token 16:    641 -> 'cc'
Token 17:  38154 -> 'ccc'
Token 18:     17 -> '2'
Token 19:  23015 -> '[N'
Token 20:     10 -> '+'
Token 21:   9725 -> ']('
Token 22:     28 -> '='
Token 23:     46 -> 'O'
Token 24:   6758 -> ')['
Token 25:     46 -> 'O'
Token 26:     12 -> '-'
Token 27:   2526 -> '])'
Token 28:     77 -> 'n'
Token 29:     17 -> '2'
Token 30:   1031 -> 'nc'
Token 31:   1031 -> 'nc'
Token 32:     17 -> '2'
Token 33:     45 -> 'N'
Token 34:     16 -> '1'
Token 35: 128257 -> '[/SMILES]'
Token 36: 128328 -> '[CONFORMER]'
Token 37:  44604 -> '[C'
Token 38:  35239 -> ']<'
Token 39:     12 -> '-'
Token 40:     17 -> '2'
Token 41:     13 -> '.'
Token 42:  20772 -> '770'
Token 43:     19 -> '4'
Token 44:   5106 -> ',-'
Token 45:     17 -> '2'
Token 46:     13 -> '.'
Token 47:  10828 -> '182'
Token 48:     24 -> '9'
Token 49:     11 -> ','
Token 50:     16 -> '1'
Token 51:     13 -> '.'
Token 52:  11483 -> '211'
Token 53:  31868 -> '>['
Token 54:     34 -> 'C'
Token 55:  35239 -> ']<'
Token 56:     12 -> '-'
Token 57:     17 -> '2'
Token 58:     13 -> '.'
Token 59:  24970 -> '047'
Token 60:     18 -> '3'
Token 61:   5106 -> ',-'
Token 62:     16 -> '1'
Token 63:     13 -> '.'
Token 64:   5926 -> '192'
Token 65:     24 -> '9'
Token 66:     11 -> ','
Token 67:     15 -> '0'
Token 68:     13 -> '.'
Token 69:  18277 -> '361'
Token 70:     23 -> '8'
Token 71:     29 -> '>'
Token 72:     16 -> '1'
Token 73:   5941 -> '=['
Token 74:     34 -> 'C'
Token 75:  35239 -> ']<'
Token 76:     12 -> '-'
Token 77:     15 -> '0'
Token 78:     13 -> '.'
Token 79:  26537 -> '849'
Token 80:     20 -> '5'
Token 81:   5106 -> ',-'
Token 82:     16 -> '1'
Token 83:     13 -> '.'
Token 84:  19838 -> '398'
Token 85:     19 -> '4'
Token 86:   5106 -> ',-'
Token 87:     15 -> '0'
Token 88:     13 -> '.'
Token 89:  12245 -> '231'
Token 90:     22 -> '7'
Token 91:   2284 -> '>('
Token 92:     58 -> '['
Token 93:     34 -> 'C'
Token 94:  35239 -> ']<'
Token 95:     12 -> '-'
Token 96:     15 -> '0'
Token 97:     13 -> '.'
Token 98:   8874 -> '124'
Token 99:     17 -> '2'

Full reconstruction matches: True

Character-to-token alignment (first 100 chars):
[SMILES]CC1=C(C(N)=O)[C@H](c2ccccc2[N+](=O)[O-])n2ncnc2N1[/SMILES][CONFORMER][C]<-2.7704,-2.1829,1.2
... (+592 more chars)

======================================================================
VALIDATION DATASET: conformers_valid
======================================================================
Dataset path: /nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v2/processed_strings/valid
Total .jsonl files: 1
Total lines in dataset: 716,818
  Randomly sampling 10000 lines from 716,818 total lines across 1 files...

FROM RANDOM SAMPLE OF 10000 LINES:
  Characters in sampled lines: 7,863,344
  Tokens produced: 5,002,964
  Samples this represents: 2,442
  Avg chars per line: 786.3
  Avg tokens per line: 500.3
  Lines per sample: 4.09

EXTRAPOLATED TO FULL DATASET:
  Estimated total characters: 563,658,651
  Estimated total tokens: 358,621,464
  Estimated total samples (seq_len=2048): 175,108

SAMPLE:
--- Sample ---
Encoded tokens (first 200 of 577): [128000, 128256, 34, 44604, 31, 39, 9725, 34, 121110, 46, 8, 45, 66, 16, 77, 641, 17, 66, 1471, 16, 8, 3791, 3100, 2432, 34, 8, 3791, 17, 28, 46, 8, 45, 16, 34, 121110, 46, 48086, 17, 641, 38154, 17, 34, 16, 28, 46, 128257, 128328, 44604, 35239, 12, 18, 13, 23428, 18, 5106, 15, 13, 26007, 19, 5106, 17, 13, 6280, 16, 31868, 34, 31, 39, 35239, 12, 17, 13, 18199, 22, 5106, 16, 13, 9992, 18, 5106, 16, 13, 12226, 20, 2284, 58, 34, 35239, 12, 17, 13, 14033, 19, 5106, 17, 13, 5162, 20, 5106, 15, 13, 14206, 16, 2284, 5941, 46, 35239, 12, 18, 13, 22266, 18, 5106, 17, 13, 24061, 17, 5106, 15, 13, 7028, 16, 29, 6758, 52371, 35239, 12, 16, 13, 24491, 18, 5106, 17, 13, 18199, 19, 11, 15, 13, 23969, 18, 31868, 66, 35239, 12, 15, 13, 18044, 23, 5106, 16, 13, 25612, 11, 15, 13, 24763, 21, 29, 16, 7824, 35239, 15, 13, 12171, 5106, 17, 13, 23785, 17, 11, 16, 13, 24242, 21, 31868, 66, 35239, 16, 13, 19608, 20, 5106, 16, 13, 15573, 24, 11, 17, 13, 25077, 23, 31868, 66, 35239, 17, 13, 9367, 18, 5106]

Decoded text:
<|begin_of_text|>[SMILES]C[C@H](C(=O)Nc1ncc2c(n1)CC(C)(C)CC2=O)N1C(=O)c2ccccc2C1=O[/SMILES][CONFORMER][C]<-3.5943,-0.9744,-2.1951>[C@H]<-2.3637,-1.1553,-1.3105>([C]<-2.6404,-2.1965,-0.2271>(=[O]<-3.6423,-2.8592,-0.1901>)[NH]<-1.6633,-2.3634,0.7363>[c]<-0.3668,-1.936,0.7926>1[n]<0.208,-2.0452,1.9936>[c]<1.4615,-1.6559,2.0838>[c]<2.1883,-1.1926,0.9852>2[c]<1.5037,-1.1544,-0.2372>([n]<0.2285,-1.5038,-0.3132>1)[C]<2.1783,-0.7409,-1.499>[C@]<3.3833,0.1716,-1.2427>([C]<2.9033,1.5335,-0.7376>)([C]<4.15...

DETAILED TOKEN ANALYSIS:
Original processed text (892 chars): [SMILES]C[C@H](C(=O)Nc1ncc2c(n1)CC(C)(C)CC2=O)N1C(=O)c2ccccc2C1=O[/SMILES][CONFORMER][C]<-3.5943,-0.9744,-2.1951>[C@H]<-2.3637,-1.1553,-1.3105>([C]<-2.6404,-2.1965,-0.2271>(=[O]<-3.6423,-2.8592,-0.190...

Token breakdown (first 100 tokens):
Token  0: 128000 -> '<|begin_of_text|>'
Token  1: 128256 -> '[SMILES]'
Token  2:     34 -> 'C'
Token  3:  44604 -> '[C'
Token  4:     31 -> '@'
Token  5:     39 -> 'H'
Token  6:   9725 -> ']('
Token  7:     34 -> 'C'
Token  8: 121110 -> '(='
Token  9:     46 -> 'O'
Token 10:      8 -> ')'
Token 11:     45 -> 'N'
Token 12:     66 -> 'c'
Token 13:     16 -> '1'
Token 14:     77 -> 'n'
Token 15:    641 -> 'cc'
Token 16:     17 -> '2'
Token 17:     66 -> 'c'
Token 18:   1471 -> '(n'
Token 19:     16 -> '1'
Token 20:      8 -> ')'
Token 21:   3791 -> 'CC'
Token 22:   3100 -> '(C'
Token 23:   2432 -> ')('
Token 24:     34 -> 'C'
Token 25:      8 -> ')'
Token 26:   3791 -> 'CC'
Token 27:     17 -> '2'
Token 28:     28 -> '='
Token 29:     46 -> 'O'
Token 30:      8 -> ')'
Token 31:     45 -> 'N'
Token 32:     16 -> '1'
Token 33:     34 -> 'C'
Token 34: 121110 -> '(='
Token 35:     46 -> 'O'
Token 36:  48086 -> ')c'
Token 37:     17 -> '2'
Token 38:    641 -> 'cc'
Token 39:  38154 -> 'ccc'
Token 40:     17 -> '2'
Token 41:     34 -> 'C'
Token 42:     16 -> '1'
Token 43:     28 -> '='
Token 44:     46 -> 'O'
Token 45: 128257 -> '[/SMILES]'
Token 46: 128328 -> '[CONFORMER]'
Token 47:  44604 -> '[C'
Token 48:  35239 -> ']<'
Token 49:     12 -> '-'
Token 50:     18 -> '3'
Token 51:     13 -> '.'
Token 52:  23428 -> '594'
Token 53:     18 -> '3'
Token 54:   5106 -> ',-'
Token 55:     15 -> '0'
Token 56:     13 -> '.'
Token 57:  26007 -> '974'
Token 58:     19 -> '4'
Token 59:   5106 -> ',-'
Token 60:     17 -> '2'
Token 61:     13 -> '.'
Token 62:   6280 -> '195'
Token 63:     16 -> '1'
Token 64:  31868 -> '>['
Token 65:     34 -> 'C'
Token 66:     31 -> '@'
Token 67:     39 -> 'H'
Token 68:  35239 -> ']<'
Token 69:     12 -> '-'
Token 70:     17 -> '2'
Token 71:     13 -> '.'
Token 72:  18199 -> '363'
Token 73:     22 -> '7'
Token 74:   5106 -> ',-'
Token 75:     16 -> '1'
Token 76:     13 -> '.'
Token 77:   9992 -> '155'
Token 78:     18 -> '3'
Token 79:   5106 -> ',-'
Token 80:     16 -> '1'
Token 81:     13 -> '.'
Token 82:  12226 -> '310'
Token 83:     20 -> '5'
Token 84:   2284 -> '>('
Token 85:     58 -> '['
Token 86:     34 -> 'C'
Token 87:  35239 -> ']<'
Token 88:     12 -> '-'
Token 89:     17 -> '2'
Token 90:     13 -> '.'
Token 91:  14033 -> '640'
Token 92:     19 -> '4'
Token 93:   5106 -> ',-'
Token 94:     17 -> '2'
Token 95:     13 -> '.'
Token 96:   5162 -> '196'
Token 97:     20 -> '5'
Token 98:   5106 -> ',-'
Token 99:     15 -> '0'

Full reconstruction matches: True

Character-to-token alignment (first 100 chars):
[SMILES]C[C@H](C(=O)Nc1ncc2c(n1)CC(C)(C)CC2=O)N1C(=O)c2ccccc2C1=O[/SMILES][CONFORMER][C]<-3.5943,-0.
... (+792 more chars)

======================================================================
SUMMARY
======================================================================

TRAIN DATASET:
  Total lines: 5,745,689
  Lines per sample: 4.10
  Estimated total tokens: 2,873,459,288
  Estimated total samples: 1,403,056

VALIDATION DATASET:
  Total lines: 716,818
  Lines per sample: 4.09
  Estimated total tokens: 358,621,464
  Estimated total samples: 175,108

======================================================================