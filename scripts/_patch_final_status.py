from pathlib import Path

p = Path('docs/d2v_attr_results.md')
t = p.read_text()

old = """* 15:41:36 **三个 job 被外部以 CANCELLED by 0 取消**（本地 watcher 未执行任何 scancel）；
* 之后重新提交全部失败：<code>sbatch: error: AssocGrpSubmitJobsLimit</code>
  （账号级提交额度被另一 session 的长期 array job 14116579 占用），因此本轮无法完成该对照；
* 已挂 <code>scripts/hpc_attr_shuffle_retry.sh</code>：每 5 分钟重试提交（最多 6 h），成功后
  自动串 extract/probe、拉回结果并写 STATUS5c；若额度释放会自动补齐本节。"""
new = """* 15:41:36 **三个 job 被外部以 CANCELLED by 0 取消**（本地 watcher 未执行任何 scancel）；
* 之后重新提交全部失败：<code>sbatch: error: AssocGrpSubmitJobsLimit</code>。到 17:07
  队列已空（14116579 已 COMPLETED），但**连一个 1 分钟、1 核的 trivial job 也无法提交**，
  说明是账号/组级提交额度或 QOS 限制而不是队列占用，需要用户或 HPC 管理员处理；
* <code>scripts/hpc_attr_shuffle_retry.sh</code> 会自动重试（每 5 分钟，最多 6 h），
  额度恢复时自动串 extract/probe、拉回结果并写 STATUS5c；本轮结束时仍被阻塞。"""
assert old in t
t = t.replace(old, new, 1)

old7 = """7. **账号提交额度（AssocGrpSubmitJobsLimit）**：shuffle 对照重提交时被账号级额度挡住
   （另一 session 的 array job 长期占用），只能每 5 分钟重试；同一时段的 3 个 shuffle job 还被
   外部 scancel（CANCELLED by 0）。这说明共享账号上跑长任务需要预留额度或错峰，是后续
   5-seed 复跑的运维前提。"""
new7 = """7. **账号提交额度（AssocGrpSubmitJobsLimit）**：shuffle 对照重提交时被账号级额度挡住；
   队列已空、连 trivial job 也被拒，说明不是队列占用，而是账号/组级 QOS 或提交限制，
   需要用户或管理员处理。同一时段的 3 个 shuffle job 还被外部 scancel（CANCELLED by 0）。
   这是后续 5-seed 复跑的运维前提：应确认账号额度或改用有额度的账号/分区。"""
assert old7 in t
t = t.replace(old7, new7, 1)
p.write_text(t)
print("results doc updated")

pp = Path('docs/plans/2026-11-24_d2v_structured_inputs_plan.md')
pt = pp.read_text()
oldp = """* **仍缺**：5 split seeds 的 CI、shuffle 负对照结果（HPC 账号提交额度被另一会话占用，
  job 被外部 scancel；重试链 scripts/hpc_attr_shuffle_retry.sh 在跑）、
  mapper-disjoint 复核、组内拆分。"""
newp = """* **仍缺**：5 split seeds 的 CI、shuffle 负对照结果（HPC 账号触发
  AssocGrpSubmitJobsLimit：队列空、连 trivial job 也无法提交，job 还被外部 scancel，
  需要账号/管理员处理；重试链 scripts/hpc_attr_shuffle_retry.sh 待额度恢复）、
  mapper-disjoint 复核、组内拆分。"""
assert oldp in pt
pt = pt.replace(oldp, newp, 1)
pp.write_text(pt)
print("plan updated")
