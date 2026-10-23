"""Merge verified literature records into a citation-ready reading ledger."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "literature"
zh = json.loads((ROOT / "chinese_candidates.json").read_text(encoding="utf-8"))["papers"]
en = json.loads((ROOT / "english_candidates.json").read_text(encoding="utf-8"))["records"]
records = zh + en
assert 15 <= len(records) <= 20
assert len({p["doi"].lower() for p in records}) == len(records)
assert all(2024 <= p["year"] <= 2026 for p in records)
assert all(p.get("paraphrase") for p in records), "Every reading entry needs source-grounded paraphrase"
intro = ["# 第一周文献清单与引用台账", "", "核验日期：2026-10-23。", "",
         f"共{len(records)}篇期刊论文：中文{len(zh)}篇、英文{len(en)}篇。近三年采用2024—2026正式卷期年份。部分论文在线首发早于2024，见各条日期说明；若导师按首次发表日期限制，需替换这些条目。",
         "中文6篇来自《软件学报》，原刊官网介绍列出CSCD及中文核心收录；学校采用的具体目录版本仍需复核。其余英文期刊不凭名称推定SCI分区或学校核心认定。",
         "多数已阅读摘要，个别有全文定向阅读；不能把本台账写成已完成18篇全文精读。可引用表述为有来源的转述草稿，不是逐字引文，正式引用需核对原文段落/页码。", "",
         "## 参考文献与原文入口", ""]
bib = []
cards = []
for n, p in enumerate(records, 1):
    authors = p["authors"]
    author_text = ", ".join(authors[:3]) + (", 等" if len(authors)>3 and p["id"].startswith("CN") else ", et al." if len(authors)>3 else "")
    journal_part = f"{p['journal']}, {p['year']}, {p.get('volume','')}"
    if p.get("issue"):
        journal_part += f"({p['issue']})"
    if p.get("pages"):
        journal_part += f": {p['pages']}"
    intro += [f"{n}. [{p['id']}] {author_text}. {p['title']}[J]. {journal_part}. DOI: {p['doi']}. [原文入口]({p['url']})", ""]
    cards += [f"### [{p['id']}] {p['title']}", "", f"- 阅读程度：{p['reading_depth']}",
              f"- 方法/研究内容：{p.get('method',p.get('summary',''))}",
              f"- 可引用转述：{p['paraphrase']}",
              f"- 论文使用位置：{p.get('thesis_use',p.get('thesis_location',''))}",
              f"- 适用范围与限制：{p.get('limitation',p.get('limitations',''))}",
              f"- 来源：{p['url']}"]
    if p.get("open_version"):
        cards.append(f"- 公开版本：{p['open_version']}")
    note = p.get("date_note")
    if not note and p.get("online_first_date"):
        note = f"在线首发 {p['online_first_date']}；正式卷期 {p.get('published_date',p['year'])}。"
    if note:
        cards.append(f"- 日期口径：{note}")
    if p.get("core_evidence_url"):
        cards.append(f"- 期刊目录证据：{p['core_evidence_url']}；以学校认可版本为准。")
    cards += ["- 后续精读记录：原文页码/章节：____；与你的数据差异：____；采用或不采用的方法：____。", ""]
    fields = {"title": p["title"], "author": " and ".join(authors), "journal": p["journal"], "year": p["year"], "volume": p.get("volume"), "number": p.get("issue"), "pages": p.get("pages"), "doi": p["doi"], "url": p["url"]}
    bib.append("@article{" + p["id"] + ",\n" + ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items() if value) + "\n}")
intro += ["## 建议阅读顺序", "", "1. 先读直接车辆研究：CN01、EN01、EN02，理解挂牌价与成交价、时间变化及不确定性。",
          "2. 再读本周模型和解释：EN03、EN04、CN03，以及EN07的数据泄漏问题。",
          "3. 然后读异常评估：EN08、EN06、CN04、CN05，设计独立人工核查及适用范围。",
          "4. 最后读区间与可视化：EN10、EN09、EN11、CN02；EN05补充长尾回归，CN06/CN07用于解释表达的扩展阅读。", "",
          "## 逐篇阅读与可引用内容", ""] + cards
(ROOT / "literature_review_ledger.md").write_text("\n".join(intro).rstrip() + "\n", encoding="utf-8")
(ROOT / "references.bib").write_text("\n\n".join(bib) + "\n", encoding="utf-8")
(ROOT / "literature_manifest.json").write_text(json.dumps({"count": len(records), "chinese":len(zh), "english":len(en), "journal_count":len(records), "unique_doi_count":len(records), "publication_years": sorted({p["year"] for p in records}), "doi": [p["doi"] for p in records]}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Verified and assembled {len(records)} unique journal references.")
