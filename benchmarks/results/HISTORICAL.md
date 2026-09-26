# 历史实验汇总 / Historical comparisons

保留原有有效实验的结论、比较数据和基本配置。逐次 JSON/CSV、trial、指纹、日志及模型产物已清理；下列汇总不构成完整原始审计档案。不同规模、模型和环境不能混合排名。当前主结论见 [TabPFN v2 报告](tabpfn-v2/report.md)。

## 已撤回结果 / Withdrawn results

旧 Online Retail、online-retail-semantic、online-retail-semantic-5seed、online-retail-semantic-retry 在切分前用全量目标分位数截尾，存在测试目标泄漏。相关准确率/语义收益结论已撤回，不继续保留无效成绩表；原文件亦已清理。修正实验保留原始正 Quantity，不做全量目标截尾。

## 历史结论 / Interpretation

- 全量 Adult/California 的受限 OpenFE 在多数组合上均值较好；搜索预算不等价，不能推断普遍优越性。
- Adult 两 seed 语义组平均 AUC 为 0.910493，描述错配为 0.910385；不足以证明语义特异性。已知 token 141,525、未知计费请求 0，保留一次无效提议的失败结论。
- 小型 Cancer/Diabetes 语义 smoke 的结果混合，记录 94,499 已知 token、一次未知计费请求；一次无效提议和一次超时保留在当时结果中，未补跑有利 seed。
- 修正后的全量 Retail、seed 22、线性模型 baseline/random RMSE 都为 71.198195，未见收益，仅作流程 smoke。
- 45 组 wide-context 离线 smoke 验证有界上下文构建；未测真实 API token、预测增益或语义优势。

## large-comparison

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["adult", "california"] |
| models | ["linear", "tree"] |
| methods | ["baseline", "random", "evolution", "openfe"] |
| seeds | [22, 33, 44] |
| trials | 4 |
| cv | 3 |
| openfe_candidates | 256 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.5.2"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | linear | baseline | 0.9087604372455876 | 0.0029279419402234302 | 3 | 1.237017097004961 | 0 |
| adult | linear | evolution | 0.9088388506793796 | 0.0028072201804866187 | 3 | 3.6992283613314307 | 0 |
| adult | linear | openfe | 0.9118153190105333 | 0.0032435329636830708 | 3 | 10.579714097000155 | 0 |
| adult | linear | random | 0.9088825460587738 | 0.0028783546344872267 | 3 | 4.802266805670418 | 0 |
| adult | tree | baseline | 0.9310272304741155 | 0.0032926488676882026 | 3 | 15.87772340233399 | 0 |
| adult | tree | evolution | 0.9307555896374602 | 0.0030635306531437115 | 3 | 36.63835050000731 | 0 |
| adult | tree | openfe | 0.9309740972366368 | 0.0027706851106649314 | 3 | 13.275967166666911 | 0 |
| adult | tree | random | 0.9307555896374602 | 0.0030635306531437115 | 3 | 52.210429499997794 | 0 |
| california | linear | baseline | 0.7113085014210728 | 0.01121863580309982 | 3 | 0.117946985992603 | 0 |
| california | linear | evolution | 0.7101185914669627 | 0.010033735252248975 | 3 | 0.2937072916683974 | 0 |
| california | linear | openfe | 0.6471240273095928 | 0.010550255380201774 | 3 | 7.432620111004023 | 0 |
| california | linear | random | 0.6935062105885058 | 0.018732288244172574 | 3 | 0.3907849163321468 | 0 |
| california | tree | baseline | 0.4547124433135059 | 0.0017884159156327775 | 3 | 5.283920847335442 | 0 |
| california | tree | evolution | 0.4545453127212556 | 0.0016397718718055901 | 3 | 11.979945986327948 | 0 |
| california | tree | openfe | 0.4324533116549994 | 0.0024169685674969585 | 3 | 8.665135902667922 | 0 |
| california | tree | random | 0.45434826953331636 | 0.001615180668423287 | 3 | 18.53490781932972 | 0 |

## covtype-openfe

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["covtype"] |
| models | ["linear"] |
| methods | ["baseline", "random", "evolution", "openfe"] |
| seeds | [22] |
| trials | 2 |
| cv | 3 |
| openfe_candidates | 32 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| covtype | linear | baseline | 0.8717445897037809 |  | 1 | 24.152606749994447 | 0 |
| covtype | linear | evolution | 0.8717273026534214 |  | 1 | 58.786616124998545 | 0 |
| covtype | linear | openfe | 0.8739558068078368 |  | 1 | 83.18787287500163 | 0 |
| covtype | linear | random | 0.8717273026534214 |  | 1 | 65.53386316700198 | 0 |

## adult-semantic

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["adult"] |
| models | ["linear"] |
| methods | ["baseline", "random", "llm", "llm_anonymous", "llm_shuffled"] |
| seeds | [22, 33] |
| trials | 3 |
| cv | 3 |
| openfe_candidates | 256 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | linear | baseline | 0.9089765377918242 | 0.004106761660254517 | 2 | 1.277686604502378 | 0 |
| adult | linear | llm | 0.910493496941213 | 0.006116544994594148 | 2 | 48.94949612500204 | 74765 |
| adult | linear | llm_anonymous | 0.9089765377918242 | 0.004106761660254517 | 2 | 42.07376412500162 | 34547 |
| adult | linear | llm_shuffled | 0.910384976924345 | 0.0059791223532564145 | 2 | 41.10454866649525 | 32213 |
| adult | linear | random | 0.9091139608652983 | 0.003912363982593347 | 2 | 4.1710428334990866 | 0 |

## public

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["cancer", "diabetes"] |
| models | ["linear", "tree"] |
| methods | ["baseline", "random", "evolution", "openfe"] |
| seeds | [11, 22, 33] |
| trials | 8 |
| cv | 3 |
| openfe_candidates | 256 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.5.2", "openfe": "0.0.12", "lightgbm": "4.7.0", "matplotlib": "3.11.2"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cancer | linear | baseline | 0.9928721174004194 | 0.004092079936256408 | 3 | 0.16812434733340828 | 0 |
| cancer | linear | evolution | 0.992732354996506 | 0.004109941566046808 | 3 | 0.8751622359995963 | 0 |
| cancer | linear | openfe | 0.9926624737945492 | 0.004192872117400354 | 3 | 5.392897347337566 | 0 |
| cancer | linear | random | 0.9920335429769392 | 0.004281043575820135 | 3 | 1.5269338333261355 | 0 |
| cancer | tree | baseline | 0.9879105520614955 | 0.007562683023602283 | 3 | 2.4383664166671224 | 0 |
| cancer | tree | evolution | 0.9881201956673656 | 0.007239031454132857 | 3 | 10.918868111334936 | 0 |
| cancer | tree | openfe | 0.9909853249475891 | 0.002362563452743121 | 3 | 5.3112573469989 | 0 |
| cancer | tree | random | 0.9877707896575821 | 0.007663709282544537 | 3 | 17.276036041332798 | 0 |
| diabetes | linear | baseline | 55.87627306692852 | 3.380440736404639 | 3 | 0.057148000332138814 | 0 |
| diabetes | linear | evolution | 56.00559212330362 | 3.369076476733786 | 3 | 0.4082688609972441 | 0 |
| diabetes | linear | openfe | 55.29887968076724 | 3.1130299400937274 | 3 | 4.5886324026699485 | 0 |
| diabetes | linear | random | 55.79130171960801 | 3.283881345731076 | 3 | 0.5799478749977425 | 0 |
| diabetes | tree | baseline | 58.48277090811757 | 3.6485248523846585 | 3 | 1.7067883333365899 | 0 |
| diabetes | tree | evolution | 58.59356375048813 | 2.2031146866937807 | 3 | 6.632785791667023 | 0 |
| diabetes | tree | openfe | 59.4783821031802 | 1.458744937868639 | 3 | 4.902029541670345 | 0 |
| diabetes | tree | random | 58.9080066246243 | 3.803170466184157 | 3 | 9.627716930340588 | 0 |

## semantic

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["cancer", "diabetes"] |
| models | ["linear"] |
| methods | ["baseline", "random", "llm", "llm_anonymous"] |
| seeds | [11, 22] |
| trials | 2 |
| cv | 3 |
| openfe_candidates | 256 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cancer | linear | baseline | 0.9907756813417191 | 0.0026683274761756504 | 2 | 0.24932175000139978 | 0 |
| cancer | linear | llm | 0.989937106918239 | 0.0038542507989203973 | 2 | 29.921626000003016 | 28069 |
| cancer | linear | llm_anonymous | 0.9909853249475891 | 0.0032612891375480824 | 2 | 53.93924310399598 | 40058 |
| cancer | linear | random | 0.990041928721174 | 0.0019271253994602182 | 2 | 0.7371301669918466 | 0 |
| diabetes | linear | baseline | 57.701913309968106 | 1.6902619213170313 | 2 | 0.07006554099643836 | 0 |
| diabetes | linear | llm | 57.24778199914556 | 2.1844963361254432 | 2 | 12.050402020497131 | 15550 |
| diabetes | linear | llm_anonymous | 57.649585907660224 | 1.6162597992894774 | 2 | 36.31914477099781 | 10822 |
| diabetes | linear | random | 57.59592836065444 | 1.5561348475855197 | 2 | 0.21403093750268454 | 0 |

## online-retail-corrected-smoke

| 配置 / Setting | 值 / Value |
| --- | --- |
| datasets | ["online_retail"] |
| models | ["linear"] |
| methods | ["baseline", "random"] |
| seeds | [22] |
| trials | 2 |
| cv | 3 |
| openfe_candidates | 256 |
| python | "3.11.14" |
| versions | {"featune": "1.0.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1"} |

| dataset | model | method | test_mean | test_std | runs | seconds_mean | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| online_retail | linear | baseline | 71.19819520436187 |  | 1 | 7.698748832990532 | 0 |
| online_retail | linear | random | 71.19819520436187 |  | 1 | 17.121663833007915 | 0 |

## wide-context-smoke

三 seed 平均值；估计 token 不等于供应商计费，离线缺失测量不补成零。

| columns | strategy | context_columns | estimated_prompt_tokens | retrieval_latency | retrieval_recall_proxy | valid_proposal_rate | prompt_tokens | test_rmse | inner_gain_per_1k_prompt_tokens | inner_gain_per_retrieved_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | full | 200.0 | 24498.0 | 0.0014762363328675665 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 200 | random | 40.0 | 7600.0 | 0.00033448633136376666 | 0.16666666666666666 | not measured | not measured | not measured | not measured | not measured |
| 200 | semantic | 40.0 | 7557.0 | 0.0058589860030527005 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 200 | semantic_grouping | 40.0 | 7416.0 | 0.004923097003484067 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 200 | semantic_memory | 40.0 | 7416.0 | 0.004912777668020334 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 500 | full | 500.0 | 56448.0 | 0.009710861340863566 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 500 | random | 40.0 | 7641.666666666667 | 0.0008479026631296333 | 0.0 | not measured | not measured | not measured | not measured | not measured |
| 500 | semantic | 40.0 | 7568.0 | 0.011226971997530167 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 500 | semantic_grouping | 40.0 | 7416.0 | 0.011106736332294467 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 500 | semantic_memory | 40.0 | 7416.0 | 0.010990291998799232 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 1000 | full | 1000.0 | 109698.0 | 0.03798491666869567 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 1000 | random | 40.0 | 7644.333333333333 | 0.0016194586642086 | 0.16666666666666666 | not measured | not measured | not measured | not measured | not measured |
| 1000 | semantic | 40.0 | 7427.0 | 0.022056958337392 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 1000 | semantic_grouping | 40.0 | 7416.0 | 0.021499194335774435 | 1.0 | not measured | not measured | not measured | not measured | not measured |
| 1000 | semantic_memory | 40.0 | 7416.0 | 0.022330096995574367 | 1.0 | not measured | not measured | not measured | not measured | not measured |
