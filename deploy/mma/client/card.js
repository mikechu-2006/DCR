import { requestFeatures } from "./core.js";

const jobs = new Map();

function elements() {
    return [document.getElementById("reform-rating"), document.getElementById("reform-value"),
        document.getElementById("reform-calculate")];
}

export function resetReformCard() {
    const [root, value, button] = elements();
    root.hidden = true;
    value.textContent = "";
    button.hidden = true;
    button.disabled = false;
    button.onclick = null;
}

export function renderReformCard(result, chart, bytes, isCurrent, refresh, canStart) {
    resetReformCard();
    if (!result || result.state === "ready") return;
    const [root, value, button] = elements();
    root.hidden = false;
    if (result.state !== "missing") {
        value.textContent = result.error || "4K or selected mods/rate unsupported";
        return;
    }

    button.hidden = false;
    button.textContent = "计算难度";
    const follow = pending => {
        button.disabled = true;
        button.textContent = "计算中…";
        pending.then(() => {
            if (isCurrent()) refresh();
        }, error => {
            if (!isCurrent()) return;
            value.textContent = error.message;
            button.disabled = false;
            button.textContent = "重试计算";
        });
    };
    button.onclick = () => {
        if (!isCurrent() || button.disabled) return;
        if (!canStart()) {
            value.textContent = "曲目结束后可计算";
            return;
        }
        const pending = requestFeatures(result, chart, bytes).finally(() => jobs.delete(result.key));
        jobs.set(result.key, pending);
        follow(pending);
    };
    if (jobs.has(result.key)) follow(jobs.get(result.key));
}
