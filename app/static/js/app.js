(() => {
  const section = document.getElementById("analyze");
  if (!section) return;

  const textarea = document.getElementById("newsInput");
  const charCount = document.getElementById("charCount");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const result = document.getElementById("result");
  const resultPanel = document.getElementById("resultPanel");
  const formError = document.getElementById("formError");
  const btnLabel = analyzeBtn.querySelector(".btn-label");
  const btnSpinner = analyzeBtn.querySelector(".btn-spinner");

  const minLength = Number(section.dataset.minLength || 20);
  const maxLength = Number(section.dataset.maxLength || 10000);

  function updateCount() {
    const n = textarea.value.length;
    charCount.textContent = `${n.toLocaleString()} / ${maxLength.toLocaleString()}`;
  }

  function setLoading(on) {
    analyzeBtn.disabled = on;
    btnSpinner.hidden = !on;
    btnLabel.textContent = on ? "Analyzing…" : "Run analysis";
  }

  function showError(message) {
    result.hidden = false;
    resultPanel.hidden = true;
    formError.hidden = false;
    formError.textContent = message;
  }

  function renderResult(data) {
    const isFake = data.label === "FAKE";
    result.hidden = false;
    formError.hidden = true;
    resultPanel.hidden = false;
    resultPanel.classList.toggle("is-fake", isFake);
    resultPanel.classList.toggle("is-real", !isFake);

    document.getElementById("resultKicker").textContent = data.verdict || "Verdict";
    document.getElementById("resultLabel").textContent = isFake ? "Likely fake" : "Likely real";
    document.getElementById("resultExplanation").textContent = data.explanation || "";
    document.getElementById("confValue").textContent = `${data.confidence}%`;
    document.getElementById("fakeValue").textContent = `${data.fake_prob}%`;
    document.getElementById("realValue").textContent = `${data.real_prob}%`;

    const fp = Number(data.fake_prob) || 0;
    const rp = Number(data.real_prob) || 0;
    document.getElementById("fakeBar").style.width = `${fp}%`;
    document.getElementById("realBar").style.width = `${rp}%`;
    document.getElementById("fakePct").textContent = `${fp}%`;
    document.getElementById("realPct").textContent = `${rp}%`;
    document.getElementById("resultMeta").textContent =
      `Model: ${data.model || "—"} · ${data.cleaned_tokens || 0} content tokens after cleaning`;

    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function analyze() {
    const text = textarea.value.trim();
    if (text.length < minLength) {
      showError(`Please enter at least ${minLength} characters.`);
      textarea.focus();
      return;
    }
    if (text.length > maxLength) {
      showError(`Text exceeds the ${maxLength.toLocaleString()}-character limit.`);
      return;
    }

    setLoading(true);
    formError.hidden = true;

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showError(data.error || `Request failed (${res.status})`);
        return;
      }
      renderResult(data);
    } catch (err) {
      showError(err.message || "Network request failed.");
    } finally {
      setLoading(false);
    }
  }

  textarea.addEventListener("input", updateCount);
  analyzeBtn.addEventListener("click", analyze);
  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    updateCount();
    result.hidden = true;
    formError.hidden = true;
    textarea.focus();
  });

  document.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      textarea.value = btn.getAttribute("data-example") || "";
      updateCount();
      textarea.focus();
    });
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && document.activeElement === textarea) {
      e.preventDefault();
      analyze();
    }
  });

  updateCount();
})();
