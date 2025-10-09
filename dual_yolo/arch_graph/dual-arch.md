# BloodScan
 Python based computer vision software project, blood test tube clamping for transportation and visual recognition for classification and detection.
白光输入 → 白光Backbone → 特征1, 特征2, 特征3
                              ↓      ↓      ↓
                          CrossTrans CrossTrans CrossTrans
                              ↑      ↑      ↑
蓝光输入 → 蓝光Backbone → 特征1, 特征2, 特征3
                              ↓
                            Neck
                              ↓
                           检测头