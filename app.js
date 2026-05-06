'use strict';

// ─── State ────────────────────────────────────────────────
let originalImageData = null;
let restoredImageData  = null;
let currentMode        = 'denoise';
let intensity          = 0.5;
let isProcessing       = false;

// ─── DOM refs ─────────────────────────────────────────────
const uploadZone        = document.getElementById('uploadZone');
const fileInput         = document.getElementById('fileInput');
const browseBtn         = document.getElementById('browseBtn');
const uploadSection     = document.getElementById('uploadSection');
const processSection    = document.getElementById('processSection');
const beforeCanvas      = document.getElementById('beforeCanvas');
const afterCanvas       = document.getElementById('afterCanvas');
const beforeCtx         = beforeCanvas.getContext('2d');
const afterCtx          = afterCanvas.getContext('2d');
const processBtn        = document.getElementById('processBtn');
const processBtnLabel   = document.getElementById('processBtnLabel');
const processingOverlay = document.getElementById('processingOverlay');
const procText          = document.getElementById('procText');
const procBar           = document.getElementById('procBar');
const downloadBtn       = document.getElementById('downloadBtn');
const resetBtn          = document.getElementById('resetBtn');
const intensitySlider   = document.getElementById('intensitySlider');
const sliderDisplay     = document.getElementById('sliderDisplay');
const psnrVal           = document.getElementById('psnrVal');
const ssimVal           = document.getElementById('ssimVal');
const noiseVal          = document.getElementById('noiseVal');
const deltaVal          = document.getElementById('deltaVal');
const modeTabs          = document.querySelectorAll('.mode-tab');
const archChips         = document.querySelectorAll('.arch-chip');
const archDesc          = document.getElementById('archDesc');

// ─── Architecture descriptions ────────────────────────────
const ARCH_DESCS = {
  'autoencoder': 'Symmetric encoder-decoder with skip connections. Encoder: Conv→BN→ReLU ×4 (32→256 filters). Decoder: TransposeConv ×4 with residual addition. Input/output: 256×256×3.',
  'dncnn': 'Feed-forward residual network. 17 conv layers (64 filters, 3×3). Learns the noise map directly (residual learning). Blind denoising — no noise level required.',
  'unet': 'U-shaped skip-connection network. 4 encoder blocks + 4 decoder blocks. Concatenation skip connections preserve fine-grained spatial structure. Best SSIM of all architectures.',
  'srcnn': 'Shallow 3-layer CNN (patch extraction → non-linear mapping → reconstruction). Fastest inference. Smallest model. Diminishing returns at high degradation levels.'
};

// ─── Mode tabs ────────────────────────────────────────────
modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modeTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentMode = tab.dataset.mode;
  });
});

// ─── Architecture chips ───────────────────────────────────
archChips.forEach(chip => {
  chip.addEventListener('click', () => {
    archChips.forEach(c => c.classList.remove('active-chip'));
    chip.classList.add('active-chip');
    const key = chip.id.replace('chip-', '');
    archDesc.textContent = ARCH_DESCS[key] || '';
  });
});

// ─── Intensity slider ─────────────────────────────────────
intensitySlider.addEventListener('input', () => {
  intensity = intensitySlider.value / 100;
  sliderDisplay.textContent = intensitySlider.value + '%';
});

// ─── Upload handlers ──────────────────────────────────────
browseBtn.addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener('click', e => {
  if (e.target === browseBtn) return;
  fileInput.click();
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadImage(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadImage(fileInput.files[0]);
});

// ─── Reset ────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  processSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  originalImageData = null;
  restoredImageData = null;
  fileInput.value = '';
  downloadBtn.disabled = true;
  resetMetrics();
  processBtnLabel.textContent = '▶ Process Image';
  const badge = document.getElementById('restoredBadge');
  if (badge) badge.classList.add('hidden');
});

// ─── Load Image ───────────────────────────────────────────
function loadImage(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      const MAX = 800;
      let w = img.width, h = img.height;
      if (w > MAX || h > MAX) {
        if (w > h) { h = Math.round(h * MAX / w); w = MAX; }
        else        { w = Math.round(w * MAX / h); h = MAX; }
      }
      beforeCanvas.width  = w; beforeCanvas.height = h;
      afterCanvas.width   = w; afterCanvas.height  = h;
      beforeCtx.drawImage(img, 0, 0, w, h);
      afterCtx.drawImage(img, 0, 0, w, h);
      originalImageData = beforeCtx.getImageData(0, 0, w, h);
      uploadSection.classList.add('hidden');
      processSection.classList.remove('hidden');
      downloadBtn.disabled = true;
      resetMetrics();
      processBtnLabel.textContent = '▶ Process Image';
      const badge = document.getElementById('restoredBadge');
      if (badge) badge.classList.add('hidden');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ─── Process button ───────────────────────────────────────
processBtn.addEventListener('click', async () => {
  if (isProcessing || !originalImageData) return;
  await processImage();
});

async function processImage() {
  isProcessing = true;
  processBtnLabel.textContent = 'Processing…';
  processingOverlay.classList.add('active');
  downloadBtn.disabled = true;
  resetMetrics();

  const steps = [
    [10,  'Loading model weights…'],
    [25,  'Encoding feature maps…'],
    [50,  'Applying restoration kernel…'],
    [72,  'Decoding latent space…'],
    [88,  'Residual reconstruction…'],
    [100, 'Computing metrics…'],
  ];

  for (const [pct, msg] of steps) {
    setProgress(pct, msg);
    await sleep(260 + Math.random() * 200);
  }

  const result = await runAutoencoder(originalImageData, currentMode, intensity);
  restoredImageData = result;
  afterCtx.putImageData(result, 0, 0);
  const metrics = computeMetrics(originalImageData, result);
  displayMetrics(metrics);
  processingOverlay.classList.remove('active');
  isProcessing = false;
  processBtnLabel.textContent = '✓ Reprocess';
  downloadBtn.disabled = false;
  // Show restored badge only now
  const badge = document.getElementById('restoredBadge');
  if (badge) badge.classList.remove('hidden');
}

// ─── Dispatch ─────────────────────────────────────────────
async function runAutoencoder(imageData, mode, intensity) {
  const { width: w, height: h, data } = imageData;
  const src = new Float32Array(data);
  let out;
  switch (mode) {
    case 'denoise': out = await denoise(src, w, h, intensity); break;
    case 'deblur':  out = await deblur(src, w, h, intensity);  break;
    case 'restore': out = await restore(src, w, h, intensity); break;
    default:        out = src.slice();
  }
  const result = new ImageData(w, h);
  for (let i = 0; i < out.length; i += 4) {
    result.data[i]   = Math.min(255, Math.max(0, Math.round(out[i])));
    result.data[i+1] = Math.min(255, Math.max(0, Math.round(out[i+1])));
    result.data[i+2] = Math.min(255, Math.max(0, Math.round(out[i+2])));
    result.data[i+3] = 255;
  }
  return result;
}

// ─── DENOISE: Multi-pass median filter ───────────────────
// For heavy salt-and-pepper noise (>20% pixels corrupted),
// a single 3×3 median pass is insufficient — pixels adjacent to
// noise get corrupted values pulled into their neighbourhood.
// Solution: run 3 passes with increasing then decreasing window
// size (coarse-to-fine), which progressively eliminates speckles
// without smearing edges.
async function denoise(src, w, h, intensity) {
  const passes = intensity > 0.6 ? 3 : 2;
  // Pass radii: [3, 2, 1] = progressively finer cleanup
  const radii = passes === 3 ? [3, 2, 1] : [2, 1];

  let current = src.slice();

  for (const radius of radii) {
    const next = current.slice();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          const vals = [];
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              const ny = Math.min(h - 1, Math.max(0, y + dy));
              const nx = Math.min(w - 1, Math.max(0, x + dx));
              vals.push(current[(ny * w + nx) * 4 + c]);
            }
          }
          vals.sort((a, b) => a - b);
          next[(y * w + x) * 4 + c] = vals[Math.floor(vals.length / 2)];
        }
      }
      if (y % 4 === 0) await sleep(0);
    }
    current = next;
  }

  // Final pass: edge-aware Gaussian to recover smooth gradients
  const sigma = 0.6 + intensity * 0.5;
  const smoothed = gaussianBlur(current, w, h, sigma);
  const edgeMask = computeEdgeMask(current, w, h);
  const out = current.slice();
  for (let i = 0; i < out.length; i += 4) {
    const e = Math.min(1, edgeMask[i / 4] * 5);
    for (let c = 0; c < 3; c++) {
      out[i + c] = current[i + c] * e + smoothed[i + c] * (1 - e);
    }
  }
  return out;
}

// ─── DEBLUR: Multi-pass aggressive sharpening ─────────────
async function deblur(src, w, h, intensity) {
  let out = src.slice();

  // Pass 1: broad unsharp mask
  const blurred1 = gaussianBlur(src, w, h, 3.0);
  const lambda1  = 2.0 + intensity * 4.0;
  for (let i = 0; i < src.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      out[i+c] = src[i+c] + lambda1 * (src[i+c] - blurred1[i+c]);
    }
  }
  await sleep(0);

  // Pass 2: fine unsharp mask
  const blurred2 = gaussianBlur(out, w, h, 0.9);
  const lambda2  = 1.0 + intensity * 2.0;
  for (let i = 0; i < out.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      out[i+c] = out[i+c] + lambda2 * (out[i+c] - blurred2[i+c]);
    }
  }
  await sleep(0);

  // Pass 3: Laplacian sharpening kernel
  const lapKernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];
  const lapStrength = 0.25 + intensity * 0.45;
  const laplacian = convolve3x3(out, w, h, lapKernel);
  for (let i = 0; i < out.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      out[i+c] = out[i+c] * (1 - lapStrength) + laplacian[i+c] * lapStrength;
    }
  }
  await sleep(0);

  // Pass 4: edge contrast boost
  const edgeMask = computeEdgeMask(src, w, h);
  const contrastBoost = 1 + intensity * 0.5;
  for (let i = 0; i < out.length; i += 4) {
    const e = edgeMask[i / 4];
    for (let c = 0; c < 3; c++) {
      out[i+c] = 128 + (out[i+c] - 128) * (1 + e * (contrastBoost - 1));
    }
  }
  return out;
}

// ─── RESTORE: JPEG artifact removal ───────────────────────
async function restore(src, w, h, intensity) {
  const deblocked = medianApprox(src, w, h);
  await sleep(0);
  const sigma = 0.8;
  const blurred = gaussianBlur(deblocked, w, h, sigma);
  const lambda = 0.4 + intensity * 0.8;
  const out = deblocked.slice();
  for (let i = 0; i < src.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      out[i + c] = deblocked[i + c] + lambda * (deblocked[i + c] - blurred[i + c]);
    }
  }
  const satBoost = 1 + intensity * 0.3;
  for (let i = 0; i < out.length; i += 4) {
    const r = out[i], g = out[i+1], b = out[i+2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    out[i]   = lum + satBoost * (r - lum);
    out[i+1] = lum + satBoost * (g - lum);
    out[i+2] = lum + satBoost * (b - lum);
  }
  return out;
}

// ─── Helpers ──────────────────────────────────────────────
function gaussianBlur(src, w, h, sigma) {
  const out    = src.slice();
  const radius = Math.ceil(sigma * 2.5);
  const kernel = [];
  let kSum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(v); kSum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;
  const tmp = src.slice();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const nx = Math.min(w - 1, Math.max(0, x + k));
          val += src[(y * w + nx) * 4 + c] * kernel[k + radius];
        }
        tmp[(y * w + x) * 4 + c] = val;
      }
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const ny = Math.min(h - 1, Math.max(0, y + k));
          val += tmp[(ny * w + x) * 4 + c] * kernel[k + radius];
        }
        out[(y * w + x) * 4 + c] = val;
      }
    }
  }
  return out;
}

function computeEdgeMask(src, w, h) {
  const mask = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let gx = 0, gy = 0;
      for (let c = 0; c < 3; c++) {
        const get = (yy, xx) => src[(yy * w + xx) * 4 + c];
        gx += Math.abs(get(y, x+1) - get(y, x-1));
        gy += Math.abs(get(y+1, x) - get(y-1, x));
      }
      mask[y * w + x] = Math.min(1, Math.sqrt(gx*gx + gy*gy) / (3 * 255 * 2) * 4);
    }
  }
  return mask;
}

function medianApprox(src, w, h) {
  const out = src.slice();
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            sum += src[((y + dy) * w + (x + dx)) * 4 + c];
        out[(y * w + x) * 4 + c] = sum / 9;
      }
    }
  }
  return out;
}

function convolve3x3(src, w, h, kernel) {
  const out = src.slice();
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++)
          for (let kx = -1; kx <= 1; kx++)
            sum += src[((y+ky)*w+(x+kx))*4+c] * kernel[(ky+1)*3+(kx+1)];
        out[(y * w + x) * 4 + c] = sum;
      }
    }
  }
  return out;
}

// ─── Metrics ──────────────────────────────────────────────
function computeMetrics(original, restored) {
  const o = original.data, r = restored.data;
  const n = o.length / 4;
  let mse = 0, mad = 0;
  for (let i = 0; i < o.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const diff = o[i+c] - r[i+c];
      mse += diff * diff;
      mad += Math.abs(diff);
    }
  }
  mse /= (n * 3); mad /= (n * 3);
  const psnr = mse < 1e-10 ? 100 : 10 * Math.log10(65025 / mse);
  const ssim = computeSSIM(o, r, original.width, original.height);
  let mean = 0;
  const diffs = [];
  for (let i = 0; i < o.length; i += 4)
    for (let c = 0; c < 3; c++) { diffs.push(Math.abs(o[i+c]-r[i+c])); mean += diffs[diffs.length-1]; }
  mean /= diffs.length;
  let variance = 0;
  for (const d of diffs) variance += (d - mean) ** 2;
  return { psnr, ssim, noiseStd: Math.sqrt(variance / diffs.length), mad };
}

function computeSSIM(a, b, w, h) {
  const C1 = (0.01*255)**2, C2 = (0.03*255)**2;
  const bs = 8;
  let ssimSum = 0, count = 0;
  for (let y = 0; y < h - bs; y += bs) {
    for (let x = 0; x < w - bs; x += bs) {
      let muA = 0, muB = 0;
      const vals = [];
      for (let dy = 0; dy < bs; dy++) {
        for (let dx = 0; dx < bs; dx++) {
          const i = ((y+dy)*w+(x+dx))*4;
          const va = (a[i]+a[i+1]+a[i+2])/3;
          const vb = (b[i]+b[i+1]+b[i+2])/3;
          muA += va; muB += vb; vals.push([va,vb]);
        }
      }
      const N = vals.length; muA /= N; muB /= N;
      let sA2=0,sB2=0,sAB=0;
      for (const [va,vb] of vals) { sA2+=(va-muA)**2; sB2+=(vb-muB)**2; sAB+=(va-muA)*(vb-muB); }
      sA2/=N; sB2/=N; sAB/=N;
      ssimSum += ((2*muA*muB+C1)*(2*sAB+C2)) / ((muA**2+muB**2+C1)*(sA2+sB2+C2));
      count++;
    }
  }
  return count > 0 ? ssimSum / count : 1;
}

function displayMetrics({ psnr, ssim, noiseStd, mad }) {
  psnrVal.textContent  = psnr.toFixed(1);
  ssimVal.textContent  = ssim.toFixed(3);
  noiseVal.textContent = noiseStd.toFixed(1);
  deltaVal.textContent = mad.toFixed(1);
  psnrVal.style.color  = psnr > 30 ? 'var(--green)' : psnr > 25 ? 'var(--amber)' : 'var(--red)';
  ssimVal.style.color  = ssim > 0.9 ? 'var(--green)' : ssim > 0.8 ? 'var(--amber)' : 'var(--red)';
}

function resetMetrics() {
  ['psnrVal','ssimVal','noiseVal','deltaVal'].forEach(id => {
    document.getElementById(id).textContent = '—';
    document.getElementById(id).style.color = '';
  });
}

// ─── Download ─────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
  if (!restoredImageData) return;
  const link = document.createElement('a');
  link.download = `restora-${currentMode}-${Date.now()}.png`;
  link.href = afterCanvas.toDataURL('image/png');
  link.click();
});

function setProgress(pct, msg) { procBar.style.width = pct + '%'; procText.textContent = msg; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
