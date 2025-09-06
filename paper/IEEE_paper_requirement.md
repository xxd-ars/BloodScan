# IEEE期刊论文撰写规范指南

## 一、文档结构和格式规范

### 1. 基本文档设置
- **文档类别**: 使用 `\documentclass[journal,twoside,web]{ieeecolor}` 用于彩色期刊
- **模板来源**: 从IEEE官方网站下载最新模板: http://ieeeauthorcenter.ieee.org/create-your-ieee-article/use-authoring-tools-and-ieee-article-templates/ieee-articletemplates/
- **页面设置**: 双面排版(twoside)，适用于网络发布(web)

### 2. 必需的包文件
```latex
\usepackage{generic}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{algorithm,algorithmic}
\usepackage{hyperref}
\usepackage{textcomp}
```

## 二、论文标题规范

### 1. 标题格式要求
- **大小写规则**: 标题应使用正确的大小写形式，而非全部大写
- **避免长公式**: 不要在标题中使用带下标的长公式，简短标识公式可以接受（如"Nd--Fe--B"）
- **不写"(Invited)"**: 标题中不要包含"(Invited)"字样

### 2. 作者信息规范
- **完整姓名**: 优先使用作者完整姓名，但非强制要求
- **姓名间距**: 作者姓名首字母之间要加空格
- **会员资格标注**: 使用 `\IEEEmembership{Fellow, IEEE}` 或 `\IEEEmembership{Member, IEEE}` 标注IEEE会员资格

## 三、摘要(Abstract)撰写规范

### 1. 长度要求
- **字数限制**: 摘要必须在150-250字之间
- **严格遵守**: 必须严格遵守字数限制，否则需要相应调整

### 2. 内容要求
- **自包含性**: 摘要必须是自包含的，不能包含缩写、脚注或参考文献
- **完整反映**: 必须简洁而全面地反映文章内容
- **微观缩影**: 摘要应该是全文的微观缩影
- **单段格式**: 必须写成单段，不能包含显示的数学方程或表格材料

### 3. 关键词要求
- **数量**: 包含3-4个不同的关键词或短语
- **目的**: 帮助读者找到文章
- **避免重复**: 避免过度重复相同短语，可能导致搜索引擎拒绝页面
- **语法正确**: 确保摘要阅读流畅且语法正确
- **IEEE主题词库**: 建议使用IEEE主题词库查找标准化关键词

## 四、文章内容撰写规范

### 1. 引言部分
- **首字母装饰**: 使用 `\IEEEPARstart{T}{his}` 创建装饰性首字母
- **模板说明**: 如果阅读的是PDF版本，建议下载LaTeX模板进行写作

### 2. 缩写和首字母缩略词
- **首次定义**: 即使在摘要中已定义，在正文中首次使用时仍需定义
- **常用缩写**: IEEE、SI、ac、dc等不需要定义
- **标点规则**: 包含句点的缩写不应有空格，如写成"C.N.R.S."而不是"C. N. R. S."
- **标题使用**: 除非不可避免（如本文标题中的"IEEE"），否则不在标题中使用缩写

### 3. 语言使用建议
- **标点间距**: 句点和冒号后使用一个空格
- **复合修饰词**: 使用连字符连接复合修饰词，如"zero-field-cooled magnetization"
- **悬垂分词**: 避免悬垂分词，如"Using (1), the potential was calculated"改为"The potential was calculated by using (1)"或"Using (1), we calculated the potential"

### 4. 数字和单位规范
- **小数点前的零**: 使用"0.25"而不是".25"
- **体积单位**: 使用"cm³"而不是"cc"
- **尺寸表示**: "0.1 cm × 0.2 cm"而不是"0.1 × 0.2 cm²"
- **时间单位**: "seconds"的缩写是"s"而不是"sec"
- **磁场单位**: 使用"Wb/m²"或"webers per square meter"而不是"webers/m²"
- **数值范围**: 写成"7 to 9"或"7--9"而不是"7∼9"

### 5. 标点和语法规范
- **括号标点**: 句末括号内容的标点在括号外（如本例）。（括号句的标点在括号内。）
- **美式英语**: 句点和逗号在引号内，如"this period."其他标点在引号外！
- **避免缩写**: 写成"do not"而不是"don't"
- **连续逗号**: 优选"A, B, and C"而不是"A, B and C"
- **人称使用**: 可以使用第一人称单数或复数，采用主动语态

### 6. 字体使用规范
- **字体限制**: 避免在同一文章中使用过多字体
- **MathJax兼容**: 记住MathJax无法处理非标准字体

## 五、公式规范

### 1. 公式编号和引用
- **连续编号**: 公式应连续编号，编号在括号内，右对齐
- **引用格式**: 使用"(1)"而不是"Eq. (1)"或"equation (1)"，除非在句首："Equation (1) is..."

### 2. 公式格式
- **符号定义**: 确保公式中的符号在公式出现前或紧随其后被定义
- **符号斜体**: 将符号斜体化（T可能指温度，但T是单位特斯拉）
- **紧凑格式**: 使用斜线(/)、exp函数或适当的指数使公式更紧凑
- **标点**: 当公式是句子的一部分时要标点

### 3. 软引用vs硬引用
- **推荐做法**: 使用"软"交叉引用如 `\eqref{Eq}` 而不是"硬"引用如 `(1)`
- **便于维护**: 便于合并章节、添加公式或改变图表、引用的顺序

### 4. 环境选择
- **避免eqnarray**: 不要使用 `{eqnarray}` 环境
- **推荐环境**: 使用 `{align}` 或 `{IEEEeqnarray}` 代替
- **原因**: `{eqnarray}` 环境在关系符号周围留有难看的空格

## 六、算法规范

### 1. 算法格式
- **编号**: 算法应该编号并包含简短标题
- **分隔线**: 用规则线将算法与文本分开，标题上下都有规则线，最后一行后也有规则线

### 2. 算法示例格式
```latex
\begin{algorithm}[H]
\caption{算法名称}\label{alg:算法标签}
\begin{algorithmic}
\STATE 算法步骤
\STATE \textbf{return} 返回值
\end{algorithmic}
\end{algorithm}
```

## 七、单位制规范

### 1. 单位制选择
- **主单位制**: 使用SI（MKS）或CGS作为主单位制（强烈建议SI）
- **英制单位**: 英制单位可作为辅助单位（在括号中）
- **数据存储**: 这适用于数据存储领域的论文
- **示例**: 写成"15 Gb/cm²（100 Gb/in²）"

### 2. 特殊情况
- **贸易标识**: 当英制单位用作贸易标识时例外，如"3½-in disk drive"
- **避免混合**: 避免混合SI和CGS单位，如电流用安培而磁场用奥斯特
- **混合单位**: 如必须使用混合单位，清楚标明方程中每个量的单位

### 3. 磁场单位
- **SI单位**: 磁场强度H的SI单位是A/m
- **特斯拉使用**: 如要使用特斯拉(T)，应指磁通密度B或磁场强度μ₀H
- **复合单位**: 使用中心点分隔复合单位，如"A·m²"

## 八、常见错误避免

### 1. 词汇使用错误
- **"data"**: "data"是复数，不是单数
- **真空磁导率**: μ₀的下标是零，不是小写字母"o"
- **剩磁术语**: "remanence"是术语，形容词是"remanent"，不要写成"remnance"或"remnant"
- **单位术语**: 使用"micrometer"而不是"micron"
- **图中图**: "inset"而不是"insert"
- **词汇选择**: "alternatively"优于"alternately"（除非指交替事件）
- **"whereas" vs "while"**: 使用"whereas"而不是"while"（除非指同时事件）
- **避免"essentially"**: 不用"essentially"表示"approximately"或"effectively"
- **避免"issue"**: 不用"issue"作为"problem"的委婉说法

### 2. 化学符号规范
- **分离符号**: 未指定组成时，用短横线分隔化学符号
- **示例**: "NiMn"表示金属间化合物Ni₀.₅Mn₀.₅，"Ni--Mn"表示某种组成的合金Ni_x Mn₁₋ₓ

### 3. 同音词辨析
- **"affect" vs "effect"**: "affect"通常是动词，"effect"通常是名词
- **"complement" vs "compliment"**: 意思不同
- **"discreet" vs "discrete"**: 意思不同
- **"principal" vs "principle"**: "principal investigator" vs "principle of measurement"
- **"imply" vs "infer"**: 不要混淆

### 4. 前缀使用规范
- **前缀连接**: "non"、"sub"、"micro"、"multi"、"ultra"等不是独立单词
- **连字符**: 应该与修饰的词连接，通常不用连字符

### 5. 拉丁缩写规范
- **"et al."**: "et"后面没有句点（也要斜体）
- **"i.e."**: 意思是"that is"（不斜体）
- **"e.g."**: 意思是"for example"（不斜体）

## 九、图表规范

### 1. 图表类型分类
#### A. 彩色/灰度图
- **用途**: 用于彩色或黑/灰阴影的图像
- **内容**: 包括照片、插图、多色图表和流程图

#### B. 线条图
- **构成**: 仅由黑色线条和形状组成
- **要求**: 不应有阴影或半色调的灰色，只有黑色和白色

#### C. 作者照片
- **规格**: 出现在论文末尾的作者头肩照

#### D. 表格
- **特点**: 通常是黑白的数据图表，有时包含彩色

### 2. 多部分图表
- **组合要求**: 由多个并排或堆叠的子图组成的图表
- **混合类型**: 如果包含多种图表类型（线条图和灰度图），应遵循更严格的指导原则

### 3. 文件格式要求
- **支持格式**: PostScript (.PS)、Encapsulated PostScript (.EPS)、Tagged Image File Format (.TIFF)、Portable Document Format (.PDF)、Portable Network Graphics (.PNG)、Metapost (.MPS)
- **提交要求**: 最终论文提交时，图形应单独以上述格式之一提交

### 4. 图表尺寸规范
- **标准宽度**: 大多数图表、图形和表格为单栏宽（3.5英寸/88毫米/21派卡）或页面宽（7.16英寸/181毫米/43派卡）
- **最大深度**: 8.5英寸（216毫米/54派卡）
- **标题空间**: 选择图形深度时，请为标题留出空间
- **灵活尺寸**: 图表可以在栏宽和页面宽之间调整大小，但建议不小于栏宽

### 5. 特殊出版物尺寸
- **IEEE Proceedings**: 栏宽为3.25英寸（82.5毫米/19.5派卡）

### 6. 作者照片尺寸
- **标准照片**: 1英寸宽×1.25英寸高（25.4×31.75毫米/6×7.5派卡）
- **社论照片**: 1.59英寸宽×2英寸高（40×50毫米/9.5×12派卡）

### 7. 分辨率要求
- **作者照片、彩色和灰度图**: 至少300dpi
- **线条图和表格**: 最少600dpi

### 8. 矢量图要求
- **支持格式**: .EPS/.PDF/.PS
- **字体要求**: 必须嵌入所有字体或将文本转换为轮廓，以获得最佳质量结果

### 9. 色彩空间
- **RGB**: 通常用于屏幕图形
- **CMYK**: 用于印刷目的
- **要求**: 所有彩色图应在RGB或CMYK色彩空间生成，灰度图应在灰度色彩空间提交

### 10. 字体要求
- **推荐字体**: Times New Roman、Helvetica、Arial、Cambria和Symbol
- **字体嵌入**: EPS、PS或PDF文件必须嵌入所有字体
- **安全选项**: 保存文件前去除字体，创建"轮廓"类型

### 11. 图表标签规范
#### A. 坐标轴标签
- **使用文字**: 使用文字而不是符号
- **示例**: 写"Magnetization"或"Magnetization M"，而不只是"M"
- **单位**: 单位放在括号中
- **避免仅单位**: 不要仅用单位标记坐标轴
- **正确格式**: "Magnetization (A/m)"或"Magnetization (A·m⁻¹)"，不只是"A/m"
- **温度示例**: "Temperature (K)"而不是"Temperature/K"

#### B. 倍数表示
- **清楚表示**: "Magnetization (kA/m)"或"Magnetization (10³ A/m)"
- **避免混淆**: 不写"Magnetization (A/m)×1000"
- **字体大小**: 图表标签应清晰，大约8到10磅字体

#### C. 多部分图表标签
- **标签格式**: 8磅Times New Roman字体，格式为(a) (b) (c)
- **位置**: 居中出现在每个子图下方

### 12. 文件命名规范
#### A. 图表命名
- **格式**: 作者姓氏前5个字母 + 顺序号
- **示例**: 作者"Anderson"的前三个图表命名为ander1.tif、ander2.tif、ander3.ps

#### B. 表格命名
- **格式**: 作者姓氏前5个字母 + .t + 表格号
- **示例**: Anderson的前三个表格命名为ander.t1.tif、ander.t2.ps、ander.t3.eps

#### C. 作者照片命名
- **格式**: 作者姓氏前5个字符
- **示例**: oppen.ps、moshc.tif、chen.eps、duran.pdf

#### D. 同名处理
- **规则**: 如果姓氏相同，用名字首字母替代第五、四、三...个字母直到区分
- **示例**: Michael和Monica Oppenheimer的照片命名为oppmi.tif和oppmo.eps

### 13. 图表引用规范
- **缩写使用**: 引用图表时使用缩写"Fig."，即使在句首
- **表格引用**: 不要缩写"Table"
- **表格编号**: 表格应使用罗马数字编号

### 14. 图表提交规范
- **位置**: 不需要将图表和表格放在每栏的顶部和底部
- **集中放置**: 所有图表、图表标题和表格可以放在论文末尾
- **单独提交**: 除了在最终手稿中提交图表外，还应单独提交图表
- **标题位置**: 图表标题放在图表下方，表格标题放在表格上方
- **避免**: 不要将标题作为图表的一部分，或将其放在链接到图表的"文本框"中
- **边框**: 不要在图表外围放置边框

### 15. IEEE期刊彩色印刷政策
- **免费彩色**: 所有IEEE期刊都允许作者在IEEE Xplore上免费发布彩色图表，并自动转换为印刷版的灰度图
- **印刷彩色**: 大多数期刊中，如作者选择，图表和表格也可以印刷成彩色，但需要额外费用
- **仅在线期刊**: 仅在线期刊的图表将免费以彩色显示

## 十、参考文献规范

### 1. 引用格式
- **行内引用**: 引用出现在行内，方括号中，标点内
- **多个引用**: 每个引用单独编号，分别用方括号
- **章节引用**: 引用书中章节时，请给出相关页码
- **引用表述**: 正文中简单提及引用号，不使用"Ref."或"reference"，除非在句首

### 2. 参考文献列表格式
- **编号位置**: 引用编号左对齐，形成独立列
- **编号格式**: 编号在方括号中
- **作者姓名**: 作者或编辑的名字缩写为首字母，首字母在姓氏前
- **作者数量**: 使用所有作者姓名；仅在未给出姓名时使用"et al"
- **会议标题**: 缩写会议标题

### 3. IEEE期刊引用要求
- **完整信息**: 提供期号、页码范围、卷号、年份和/或月份（如可获得）
- **专利引用**: 引用专利时，提供发布或申请的日期和月份
- **信息完整性**: 引用可能不包含所有信息；请获取并包含相关信息
- **不合并引用**: 不要合并引用，每个编号只能有一个引用
- **URL信息**: 如果印刷引用包含URL，可以包含在引用末尾

### 4. 标题格式规范
- **书籍外**: 除书籍外，论文标题仅首词大写，专有名词和元素符号除外
- **翻译期刊**: 对于翻译期刊发表的论文，先给英文引用，再给原外文引用

### 5. 基本引用格式

#### A. 书籍格式
```
J. K. Author, "Title of chapter in the book," in Title of His Published Book, xth ed. City of Publisher, (only U.S. State), Country: Abbrev. of Publisher, year, ch. x, sec. x, pp. xxx–xxx.
```

#### B. 期刊格式
```
J. K. Author, "Name of paper," Abbrev. Title of Periodical, vol. x, no. x, pp. xxx-xxx, Abbrev. Month, year, DOI. 10.1109.XXX.123456.
```

#### C. 报告格式
```
J. K. Author, "Title of report," Abbrev. Name of Co., City of Co., Abbrev. State, Country, Rep. xxx, year.
```

#### D. 手册格式
```
Name of Manual/Handbook, x ed., Abbrev. Name of Co., City of Co., Abbrev. State, Country, year, pp. xxx-xxx.
```

#### E. 在线书籍格式
```
J. K. Author, "Title of chapter in the book," in Title of Published Book, xth ed. City of Publisher, State, Country: Abbrev. of Publisher, year, ch. x, sec. x, pp. xxx–xxx. [Online]. Available: http://www.web.com
```

#### F. 在线期刊格式
```
J. K. Author, "Name of paper," Abbrev. Title of Periodical, vol. x, no. x, pp. xxx-xxx, Abbrev. Month, year. Accessed on: Month, Day, year, doi: 10.1109.XXX.123456, [Online].
```

#### G. 会议论文格式（已发表）
```
J. K. Author, "Title of paper," in Abbreviated Name of Conf., City of Conf., Abbrev. State (if given), Country, year, pp. xxxxxx.
```

#### H. 专利格式
```
J. K. Author, "Title of patent," U.S. Patent x xxx xxx, Abbrev. Month, day, year.
```

#### I. 学位论文格式
```
a) J. K. Author, "Title of thesis," M.S. thesis, Abbrev. Dept., Abbrev. Univ., City of Univ., Abbrev. State, year.
b) J. K. Author, "Title of dissertation," Ph.D. dissertation, Abbrev. Dept., Abbrev. Univ., City of Univ., Abbrev. State, year.
```

#### J. 标准格式
```
a) Title of Standard, Standard number, date.
b) Title of Standard, Standard number, Corporate author, location, date.
```

#### K. 数据集格式
```
Author, Date, Year. "Title of Dataset," distributed by Publisher/Distributor, http://url.com (or if DOI is used, end with a period)
```

#### L. 代码格式
```
Author, Date published or disseminated, Year. "Complete title, including ed./vers.#," distributed by Publisher/Distributor, http://url.com (or if DOI is used, end with a period)
```

## 十一、脚注规范

### 1. 脚注编号
- **单独编号**: 脚注单独编号，使用上标数字
- **建议**: 建议避免脚注（除首页不编号的接收日期脚注外）
- **整合建议**: 尝试将脚注信息整合到正文中

### 2. 脚注位置
- **页面底部**: 将实际脚注放在引用列的底部
- **不放参考文献**: 不要将脚注放在参考文献列表中（尾注）
- **表格脚注**: 表格脚注使用字母

## 十二、提交和发表规范

### 1. 审稿阶段（IEEE ScholarOne Manuscripts）
- **电子提交**: 期刊、杂志和快报的稿件可通过IEEE ScholarOne Manuscripts电子提交
- **选择出版物**: 使用工具选择正确的出版物和同行评议网站
- **账户要求**: 可用IEEE网络账户登录，没有账户需创建新账户
- **提交流程**: 在作者中心点击"Start New Submission"
- **手稿类型**: 需从期刊预定列表选择手稿类型
- **完整提交**: 每步必须点击"Save and Continue"，仅上传论文不足够
- **确认**: 最后应看到提交完成确认并收到邮件确认

### 2. 最终提交阶段
- **特殊说明**: 接受后将收到关于最终文件提交的特殊说明邮件
- **避免延误**: 遵循说明避免发表延误
- **提交方式**: 大多数期刊要求通过IEEE ScholarOne Manuscripts上传最终提交
- **文件要求**: 最终提交应包括接受手稿的源文件、高质量图形文件和格式化PDF文件

### 3. 联系信息要求
- **完整信息**: 上传包含所有作者完整联系信息的文件
- **信息内容**: 包括完整邮寄地址、电话号码、传真号码和电子邮件地址
- **通信作者**: 指定在IEEE ScholarOne Manuscripts提交手稿的作者为"通信作者"
- **校样发送**: 只有通信作者会收到论文校样

### 4. 版权表格
- **电子表格**: 作者必须在提交最终手稿文件时提交电子IEEE版权表格(eCF)
- **访问方式**: 可通过手稿提交系统或作者门户访问eCF系统
- **必要批准**: 负责获取任何必要的批准和/或安全许可
- **知识产权**: 访问IEEE知识产权部门网页获取额外知识产权信息

## 十三、IEEE出版政策

### 1. 原创性要求
- **原创作品**: 作者只能提交既未在其他地方发表过，也未在其他同行评议出版物审查中的原创作品
- **披露要求**: 提交作者必须在提交手稿时披露所有先前出版物和当前提交
- **初步数据**: 不要发表"初步"数据或结果
- **作者责任**: 提交作者负责在提交文章前获得所有合著者的同意和雇主或赞助商的必要同意

### 2. 作者资格
- **强烈反对**: IEEE期刊和杂志部门强烈反对礼貌性作者资格
- **引用义务**: 作者有义务仅引用相关的先前工作

### 3. 会议相关政策
- **不发表**: IEEE期刊和杂志部门不发表会议记录或论文集
- **相关文章**: 可以发表与经过严格同行评议的会议相关的文章
- **最少评议**: 每篇提交同行评议的文章至少需要两个评议

## 十四、致谢规范

### 1. 拼写规范
- **美式英语**: "acknowledgment"的首选拼写是"g"后面没有"e"
- **标题形式**: 即使有多个致谢，也使用单数标题

### 2. 表达方式
- **避免表达**: 避免"One of us (S.B.A.) would like to thank..."的表达
- **推荐表达**: 使用"F. A. Author thanks..."
- **资助致谢**: 大多数情况下，赞助商和财务支持致谢放在首页无编号脚注中，而不是这里

## 十五、作者简介规范

### 1. 简介内容结构
#### 第一段
- **出生信息**: 可包含出生地点和/或日期（先地点，后日期）
- **教育背景**: 列出学历，包括学位类型、专业领域、机构、城市、州、国家和获得年份
- **专业领域**: 主要研究领域应小写

#### 第二段
- **人称使用**: 使用人称代词（他或她），不使用作者姓氏
- **工作经历**: 列出军事和工作经历，包括暑期和奖学金工作
- **职位大写**: 职位标题要大写
- **当前工作**: 当前工作必须有地点；以前职位可不列地点
- **出版信息**: 可包含先前出版物信息
- **出版限制**: 不要列出超过三本书或发表的文章
- **书籍格式**: 简介中书籍出版商格式：书名（出版商名称，年份）类似于引用
- **研究兴趣**: 当前和以前的研究兴趣结束本段

#### 第三段
- **开头**: 以作者头衔和姓氏开头（如Dr. Smith, Prof. Jones, Mr. Kajor, Ms. Hunter）
- **专业协会**: 列出除IEEE外的专业协会会员资格
- **奖项和工作**: 列出IEEE委员会和出版物的任何奖项和工作

### 2. 照片要求
- **质量**: 如提供照片，应质量良好，看起来专业
- **尺寸**: 参见前面章节的具体尺寸要求

### 3. 无照片简介
- **格式**: 使用 `\begin{IEEEbiographynophoto}{Author Name}` 环境
- **标准文本**: "photograph and biography not available at the time of publication."

## 十六、LaTeX特定建议

### 1. 交叉引用
- **软引用**: 使用"软"交叉引用（如 `\eqref{Eq}`）而不是"硬"引用（如`(1)`）
- **便于维护**: 便于合并章节、添加方程或改变图表、引用顺序

### 2. 数学环境
- **避免eqnarray**: 不要使用 `{eqnarray}` 方程环境
- **推荐环境**: 使用 `{align}` 或 `{IEEEeqnarray}` 代替
- **空格问题**: `{eqnarray}` 环境在关系符号周围留有难看的空格

### 3. 子方程环境注意事项
- **计数器增加**: `{subequations}` 环境即使没有显示方程编号也会增加主方程计数器
- **编号跳跃**: 忘记这点可能导致方程编号从(17)跳到(20)

### 4. BibTeX使用
- **数据来源**: BibTeX从.bib文件获取文献数据
- **文件提交**: 如使用BibTeX制作参考文献，必须发送.bib文件

### 5. 标签使用注意事项
- **避免重复**: 如给子小节和表格分配相同标签，可能发现Table I被交叉引用为Table IV-B3
- **位置重要**: 不要在更新计数器的命令之前放 `\label` 命令
- **图表标签**: `\label` 命令不应放在图表或表格的标题之前

### 6. 数组环境
- **避免nonumber**: 不要在 `{array}` 环境中使用 `\nonumber`
- **不会停止**: 它不会停止 `{array}` 内的方程编号（无论如何都没有），可能会停止周围方程中需要的方程编号

### 7. 彩色期刊设置
- **文档类**: 对于彩色期刊，使用以下设置确保外观类似最终版本：
```latex
\documentclass[journal,twoside,web]{ieeecolor}
\usepackage{Journal_Name}
```

*本指南基于IEEE期刊论文模板总结，旨在帮助作者规范地撰写IEEE期刊论文。请在实际使用时参考最新的IEEE官方指导文件。*