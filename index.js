let model;
let wordIndex = {};
let indexWord = {};

const SEQ_LEN = 20;
let running = false;

// ======================
// UI HELPERS
// ======================
function status(msg) {
    document.getElementById("status").textContent = msg;
}

function outputPredictions(preds) {
    const container = document.getElementById("output");
    container.innerHTML = "";

    preds.forEach(p => {
        const btn = document.createElement("button");
        btn.textContent = `${p.word} (${(p.prob * 100).toFixed(2)}%)`;
        btn.style.margin = "4px";

        btn.onclick = async () => {
            appendWord(p.word);
            await predictAndRender();
        };

        container.appendChild(btn);
    });
}

function getPrompt() {
    return document.getElementById("prompt").value.trim();
}

function setPrompt(text) {
    document.getElementById("prompt").value = text;
}

function appendWord(word) {
    const current = getPrompt();
    setPrompt((current + " " + word).trim());
}

// ======================
// TOKENIZATION
// ======================
function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^a-zäöüß0-9\s]/gi, " ")
        .split(/\s+/)
        .filter(Boolean);
}

function textToSequence(text) {
    const words = tokenize(text).slice(-SEQ_LEN);
    const padded = Array(SEQ_LEN).fill(0);

    words.forEach((word, i) => {
        padded[SEQ_LEN - words.length + i] = wordIndex[word] ?? 0;
    });

    return padded;
}

// ======================
// MODEL LOADING
// ======================
async function loadModel() {
    status("Lade Modell...");

    const modelUrl = "./tfjs_model/model.json";
    model = await tf.loadLayersModel(modelUrl);

    if (!model || typeof model.predict !== "function") {
        throw new Error("Modell konnte nicht geladen werden");
    }

    status("Modell bereit.");
}

// ======================
// TOKENIZER
// ======================
async function loadTokenizer() {
    status("Lade Tokenizer...");

    const res = await fetch("./tfjs_model/tokenizer_word_index.json");
    const data = await res.json();

    wordIndex = (data.word_index) ? data.word_index : data;

    indexWord = {};
    Object.entries(wordIndex).forEach(([word, idx]) => {
        indexWord[idx] = word;
    });

    status("Tokenizer geladen.");
}

// ======================
// PREDICTION
// ======================
async function predictNext(promptText, topK = 10) {
    const seq = textToSequence(promptText);
    const input = tf.tensor2d([seq], [1, SEQ_LEN], "int32");

    const pred = model.predict(input);
    const probs = await pred.data();

    input.dispose();
    pred.dispose();

    const OOV_WORD = "<OOV>";

    const filtered = Array.from(probs)
        .map((p, i) => {
            const word = indexWord[i];

            if (!word || word === OOV_WORD || word === "<unk>") return null;

            return { word, prob: p };
        })
        .filter(Boolean)
        .sort((a, b) => b.prob - a.prob)
        .slice(0, topK);

    return filtered;
}

function sampleFromPredictions(preds) {
    // Summe der Wahrscheinlichkeiten
    const total = preds.reduce((sum, p) => sum + p.prob, 0);

    let r = Math.random() * total;

    for (const p of preds) {
        r -= p.prob;
        if (r <= 0) {
            return p;
        }
    }

    return preds[preds.length - 1];
}

// ======================
// RENDER
// ======================
async function predictAndRender() {
    const prompt = getPrompt();

    if (!prompt) {
        status("Bitte Text eingeben.");
        return;
    }

   // status("Berechne Vorhersage...");

    const preds = await predictNext(prompt, 10);

    outputPredictions(preds);

    status("Vorhersage fertig.");
}

// ======================
// BUTTONS
// ======================
document.getElementById("predict").onclick = predictAndRender;

document.getElementById("next").onclick = async () => {
    const prompt = getPrompt();
    if (!prompt) return;

    const preds = await predictNext(prompt, 1);
    appendWord(preds[0].word);
    await predictAndRender();
};

document.getElementById("auto").onclick = async () => {
    running = true;

    for (let i = 0; i < 10; i++) {
        if (!running) break;

        const prompt = getPrompt();

        // Top-10 holen
        const preds = await predictNext(prompt, 10);

        // Zufällig nach Wahrscheinlichkeit auswählen
        const next = sampleFromPredictions(preds);

        appendWord(next.word);

        await predictAndRender();

        await new Promise(r => setTimeout(r, 400));
    }
};

document.getElementById("stop").onclick = () => {
    running = false;
};

document.getElementById("reset").onclick = () => {
    running = false;
    setPrompt("");
    document.getElementById("output").innerHTML = "";
    status("Zurückgesetzt.");
};

// ======================
// 📊 EVALUATION + PLOTLY
// ======================
function drawPlotlyChart(data) {
    const topk = data.topk_accuracy;
    const isDark = document.body.classList.contains("dark");

    Plotly.newPlot("plotly-chart", [{
        x: Object.keys(topk),
        y: Object.values(topk),
        type: "bar"
    }], {
        title: "Top-k Accuracy",
        paper_bgcolor: isDark ? "#222" : "#fff",
        plot_bgcolor: isDark ? "#222" : "#fff",
        font: { color: isDark ? "#fff" : "#000" }
    });
}

fetch("./evaluation_result.json")
    .then(res => res.json())
    .then(data => {
        const topk = data.topk_accuracy;

        document.getElementById("k1").textContent = topk[1];
        document.getElementById("k5").textContent = topk[5];
        document.getElementById("k10").textContent = topk[10];
        document.getElementById("k20").textContent = topk[20];

     //   document.getElementById("perplexity").textContent = data.perplexity;

        drawPlotlyChart(data);
    });

// ======================
// INIT
// ======================
window.onload = async () => {
    await loadTokenizer();
    await loadModel();
    status("Bereit.");
};