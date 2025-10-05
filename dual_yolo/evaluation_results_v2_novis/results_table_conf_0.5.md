# Evaluation Results (Confidence = 0.5)

| Method | Serum/Plasma | | | | Buffy Coat | | | |
|--------|--------------|----|----|----|-----------|----|----|----|
| | Detection Rate | IOU | Diff_up | Diff_low | Detection Rate | IOU | Diff_up | Diff_low |
|--------|----------------|-----|---------|----------|----------------|-----|---------|----------|
| id | 100.00 | 0.97$\pm$0.01 | 5.4$\pm$5.8 | \underline{3.2$\pm$3.3} | 100.00 | 0.77$\pm$0.07 | 4.3$\pm$3.1 | 2.2$\pm$1.5 |
| id-blue | 100.00 | \underline{0.98$\pm$0.01} | 4.2$\pm$3.0 | 3.4$\pm$2.4 | \textbf{100.00} | \textbf{0.77$\pm$0.07} | 4.3$\pm$3.0 | 2.4$\pm$1.7 |
| id-white | 100.00 | 0.97$\pm$0.01 | 4.3$\pm$3.7 | 3.6$\pm$2.9 | \underline{100.00} | 0.75$\pm$0.08 | 4.6$\pm$3.6 | 2.6$\pm$2.2 |
| id-blue-30 | 99.72 | 0.98$\pm$0.01 | 4.1$\pm$3.2 | 3.6$\pm$2.6 | 99.72 | \underline{0.77$\pm$0.07} | 4.3$\pm$2.9 | 2.5$\pm$1.9 |
| id-white-30 | \textbf{100.00} | 0.97$\pm$0.01 | \underline{3.9$\pm$3.8} | 3.7$\pm$2.9 | 99.72 | 0.75$\pm$0.07 | 4.5$\pm$3.3 | 2.5$\pm$1.9 |
| concat-compress | 99.44 | 0.97$\pm$0.02 | 4.4$\pm$4.1 | 4.3$\pm$6.1 | 90.00 | 0.76$\pm$0.06 | 4.2$\pm$2.5 | 2.1$\pm$1.4 |
| weighted-fusion | 0.00 | 0.00 | \textbf{0.0} | \textbf{0.0} | 71.39 | 0.76$\pm$0.06 | \textbf{3.5$\pm$2.2} | 2.2$\pm$1.5 |
| crossattn | 53.06 | 0.97$\pm$0.01 | 7.0$\pm$13.9 | 4.5$\pm$5.8 | 96.94 | 0.76$\pm$0.07 | 4.3$\pm$2.7 | \underline{2.0$\pm$1.4} |
| crossattn-30epoch | \underline{100.00} | \textbf{0.98$\pm$0.01} | 4.7$\pm$4.5 | 3.6$\pm$2.8 | 99.44 | 0.76$\pm$0.07 | 4.6$\pm$3.2 | 2.6$\pm$2.0 |
| crossattn-precise | 46.39 | 0.96$\pm$0.03 | 6.8$\pm$13.7 | 4.6$\pm$5.2 | 86.67 | 0.76$\pm$0.06 | \underline{3.8$\pm$2.5} | \textbf{1.8$\pm$1.3} |


**Note**: **Bold** = Best, _Italic_ = Second Best
