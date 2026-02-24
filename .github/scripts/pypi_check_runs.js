"use strict";

const WORKFLOW_RUN_ID_PATTERN = /\/actions\/runs\/(\d+)(?:\/|$)/;

async function listCheckRunsForRef(github, { owner, repo, ref }) {
  return (
    await github.paginate(
      github.rest.checks.listForRef,
      { owner, repo, ref, per_page: 100 },
      (response) => (Array.isArray(response.data) ? response.data : (response.data.check_runs || []))
    )
  ).filter(Boolean);
}

function getDetailsUrl(run) {
  return run?.details_url ?? "";
}

function extractWorkflowRunId(detailsUrl) {
  const match = String(detailsUrl || "").match(WORKFLOW_RUN_ID_PATTERN);
  return match ? Number(match[1]) : null;
}

module.exports = {
  WORKFLOW_RUN_ID_PATTERN,
  listCheckRunsForRef,
  getDetailsUrl,
  extractWorkflowRunId,
};
