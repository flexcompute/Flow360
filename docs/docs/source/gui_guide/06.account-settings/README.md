# Account Settings

*Manage your personal Flow360 account preferences, authentication, and security settings.*

---

## Available Pages

| *Page* | *Description* |
|--------|---------------|
| [Preferences](./01.preferences.md) | Configure default project loading behaviour and default mesher selection |
| [Virtual GPU Scheduler](./02.virtual-gpu-scheduler.md) | Monitor daily vGPU allocation, view and manage the job queue, adjust job priorities, and switch jobs to FlexCredits billing |
| [Billing](./03.billing.md) | View your Flex Credit balance, company-wide storage usage, transaction history, and per-user usage breakdown on one page |

---

## Detailed Descriptions

### [Preferences](./01.preferences.md)

*The Preferences page lets you configure account-level defaults that apply whenever you create or open a project.*

**Settings covered:**
- Load project graphics resolution (low-resolution first, high-resolution first, high-resolution only, low-resolution only)
- Default mesher selection (Legacy mesher, Beta mesher, GeometryAI mesher)

---

### [Virtual GPU Scheduler](./02.virtual-gpu-scheduler.md)

*The Virtual GPU Scheduler tab lets you monitor your daily vGPU allocation, track the remaining run time, and manage all jobs currently queued or running under your vGPU license.*

- **Overview Metrics:** Daily Total Run Time, Remaining Run Time, Reset Timer, and vGPU slot usage at a glance.
- **vGPU Usage Table:** Lists all queued and running jobs with user, submit time, resource name, priority, and status.
- **Job Actions:** Change scheduling priority, switch a job to FlexCredits billing, or delete a case from the context menu (⋮).
- **Queue Capacity:** When all slots are occupied, new jobs queue automatically; switch to FlexCredits for immediate execution.

---

### [Billing](./03.billing.md)

*The Billing page brings your Flex Credit balance, storage usage, transaction history, and per-user usage breakdown together on a single screen.*

- **FlexCredits:** Balance and expiration, company-wide storage usage, a billing breakdown by period, credit injection history, and per-user monthly usage (with CSV export).
- **Enterprise account:** Company-wide transaction history (administrators only).
- **Individual account:** Your personal transaction history (labelled **Transactions** for non-administrators).
- **My expenses:** A pie chart and table of your monthly expenses, split by simulation, storage, and support engineering fees.

---

```{toctree}
:hidden:
:maxdepth: 2
./01.preferences.md
./02.virtual-gpu-scheduler.md
./03.billing.md
```
