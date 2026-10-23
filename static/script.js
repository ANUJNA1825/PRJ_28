const dropzone = document.getElementById('dropzone');
const input = document.getElementById('image-input');
const preview = document.getElementById('preview');
const form = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');

const resultEmpty = document.getElementById('result-empty');
const resultContent = document.getElementById('result-content');
const resultBadge = document.getElementById('result-badge');
const resultLabel = document.getElementById('result-label');
const probList = document.getElementById('prob-list');

// --- drag & drop styling ---
['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  })
);

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    input.files = e.dataTransfer.files;
    showPreview(file);
  }
});

input.addEventListener('change', () => {
  if (input.files[0]) showPreview(input.files[0]);
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// --- submit for prediction ---
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!input.files[0]) {
    alert('Please choose an image first.');
    return;
  }

  submitBtn.textContent = 'Analyzing…';
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('image', input.files[0]);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.error || 'Prediction failed.');
      return;
    }

    renderResult(data);
  } catch (err) {
    alert('Something went wrong reaching the server.');
    console.error(err);
  } finally {
    submitBtn.textContent = 'Predict pollution level';
    submitBtn.disabled = false;
  }
});

function renderResult(data) {
  resultEmpty.style.display = 'none';
  resultContent.style.display = 'block';

  resultBadge.className = `result-badge ${data.predicted_class}`;
  resultLabel.textContent = `${data.predicted_class} pollution — ${data.confidence}% confidence`;

  probList.innerHTML = '';
  Object.entries(data.probabilities).forEach(([cls, pct]) => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <div class="prob-head">
        <span>${cls}</span>
        <span class="mono">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${cls}" style="width:${pct}%"></div>
      </div>
    `;
    probList.appendChild(row);
  });
}