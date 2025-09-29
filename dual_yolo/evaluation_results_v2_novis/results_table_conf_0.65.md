# Evaluation Results (Confidence = 0.65)

| Method | Serum/Plasma | | | | Buffy Coat | | | |
|--------|--------------|----|----|----|-----------|----|----|----|
| | Detection Rate | IOU | Diff_up | Diff_low | Detection Rate | IOU | Diff_up | Diff_low |
|--------|----------------|-----|---------|----------|----------------|-----|---------|----------|
| id-blue | 99.72 | \underline{0.98$\pm$0.01} | 4.1$\pm$3.2 | \underline{3.6$\pm$2.6} | \textbf{99.17} | \underline{0.78$\pm$0.07} | 4.3$\pm$2.9 | \underline{2.5$\pm$1.9} |
| id-white | \underline{100.00} | 0.97$\pm$0.01 | \textbf{3.9$\pm$3.8} | 3.7$\pm$2.9 | \underline{98.61} | 0.75$\pm$0.07 | 4.6$\pm$3.3 | 2.5$\pm$1.9 |
| concat-compress | 99.72 | 0.97$\pm$0.01 | 4.4$\pm$3.1 | 4.3$\pm$4.2 | 68.89 | 0.77$\pm$0.06 | \textbf{3.8$\pm$2.2} | 2.6$\pm$1.8 |
| weighted-fusion | 99.17 | 0.97$\pm$0.02 | 4.5$\pm$14.5 | 3.6$\pm$2.8 | 94.72 | 0.76$\pm$0.07 | \underline{4.0$\pm$2.3} | 2.7$\pm$1.9 |
| crossattn | 97.50 | 0.97$\pm$0.01 | \underline{4.1$\pm$4.3} | 5.4$\pm$7.0 | 6.11 | 0.77$\pm$0.06 | 4.2$\pm$2.5 | \textbf{2.4$\pm$1.2} |
| crossattn-30epoch | \textbf{100.00} | \textbf{0.98$\pm$0.01} | 4.7$\pm$4.5 | \textbf{3.6$\pm$2.8} | 97.50 | 0.76$\pm$0.07 | 4.6$\pm$3.0 | 2.6$\pm$2.0 |
| crossattn-precise | 91.67 | 0.97$\pm$0.01 | 4.8$\pm$4.4 | 4.0$\pm$5.2 | 0.56 | \textbf{0.80$\pm$0.02} | 6.4$\pm$1.3 | 4.0$\pm$1.5 |


**Note**: **Bold** = Best, _Italic_ = Second Best
