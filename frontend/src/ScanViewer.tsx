import { useEffect, useRef, useState, type RefObject } from "react";
import { AlertCircle, FolderOpen, Layers3, RotateCcw, Upload } from "lucide-react";
import { IS_HOSTED_DEMO, type Analysis } from "./api";
import { drawSlice, type Volume } from "./nifti";

type Props = {
  file: File | null;
  volume: Volume | null;
  result: Analysis | null;
  busy: boolean;
  fileInput: RefObject<HTMLInputElement>;
  onPick: (file?: File) => void;
  onClear: () => void;
  patientMode?: boolean;
};
type Plane = "Axial" | "Coronal" | "Sagittal";
type ViewMode = "original" | "segmentation" | "mask" | "heatmap" | "compare";
const planes: Plane[] = ["Axial", "Coronal", "Sagittal"];

type DynamicMask = { maskUrl: string; overlayUrl: string; positivePixels: number; totalPixels: number; componentCount: number };

async function renderThresholdMask(probabilityBase64: string, threshold: number, minComponentArea: number): Promise<DynamicMask> {
  const image = new Image();
  image.src = `data:image/png;base64,${probabilityBase64}`;
  await image.decode();
  const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const pixels = width * height;
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("Could not read the probability preview.");
  sourceContext.drawImage(image, 0, 0, width, height);
  const source = sourceContext.getImageData(0, 0, width, height).data;
  const candidate = new Uint8Array(pixels);
  for (let index = 0; index < pixels; index += 1) candidate[index] = source[index * 4] / 255 >= threshold ? 1 : 0;

  const visited = new Uint8Array(pixels);
  const kept = new Uint8Array(pixels);
  const queue = new Int32Array(pixels);
  let componentCount = 0;
  let positivePixels = 0;
  for (let seed = 0; seed < pixels; seed += 1) {
    if (!candidate[seed] || visited[seed]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    visited[seed] = 1;
    while (head < tail) {
      const pixel = queue[head++];
      const x = pixel % width;
      const y = (pixel / width) | 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        const nextY = y + dy;
        if (nextY < 0 || nextY >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nextX = x + dx;
          if (nextX < 0 || nextX >= width) continue;
          const next = nextY * width + nextX;
          if (candidate[next] && !visited[next]) {
            visited[next] = 1;
            queue[tail++] = next;
          }
        }
      }
    }
    if (tail >= minComponentArea) {
      componentCount += 1;
      positivePixels += tail;
      for (let index = 0; index < tail; index += 1) kept[queue[index]] = 1;
    }
  }

  const maskCanvas = document.createElement("canvas");
  const overlayCanvas = document.createElement("canvas");
  maskCanvas.width = overlayCanvas.width = width;
  maskCanvas.height = overlayCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d");
  const overlayContext = overlayCanvas.getContext("2d");
  if (!maskContext || !overlayContext) throw new Error("Could not render the thresholded mask.");
  const maskData = maskContext.createImageData(width, height);
  const overlayData = overlayContext.createImageData(width, height);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const offset = pixel * 4;
    const value = kept[pixel] ? 255 : 0;
    maskData.data[offset] = value;
    maskData.data[offset + 1] = value;
    maskData.data[offset + 2] = value;
    maskData.data[offset + 3] = 255;
    if (kept[pixel]) {
      overlayData.data[offset] = 225;
      overlayData.data[offset + 1] = 72;
      overlayData.data[offset + 2] = 83;
      overlayData.data[offset + 3] = 255;
    }
  }
  maskContext.putImageData(maskData, 0, 0);
  overlayContext.putImageData(overlayData, 0, 0);
  return {
    maskUrl: maskCanvas.toDataURL("image/png"),
    overlayUrl: overlayCanvas.toDataURL("image/png"),
    positivePixels,
    totalPixels: pixels,
    componentCount,
  };
}

export function ScanViewer({ file, volume, result, busy, fileInput, onPick, onClear, patientMode = false }: Props) {
  const [plane, setPlane] = useState<Plane>("Axial");
  const [slice, setSlice] = useState(0);
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("original");
  const [opacity, setOpacity] = useState(.46);
  const [maskThreshold, setMaskThreshold] = useState(.5);
  const [minComponentArea, setMinComponentArea] = useState(0);
  const [dynamicMask, setDynamicMask] = useState<DynamicMask | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const isVolume = Boolean(file?.name.toLowerCase().endsWith(".nii") || file?.name.toLowerCase().endsWith(".nii.gz"));
  const imageLike = Boolean(file && !isVolume);
  const maxSlice = volume ? Math.max(0, (plane === "Axial" ? volume.dims[2] : plane === "Coronal" ? volume.dims[1] : volume.dims[0]) - 1) : 0;
  const source = result?.source_png_base64 ? `data:image/png;base64,${result.source_png_base64}` : preview;
  const segmentation = result?.segmentation;
  const explanation = result?.explanation;

  useEffect(() => {
    if (file && imageLike) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreview(null);
  }, [file, imageLike]);
  useEffect(() => {
    if (volume && canvas.current) drawSlice(canvas.current, volume, plane, slice);
  }, [volume, plane, slice]);
  useEffect(() => {
    if (segmentation) {
      setMode("segmentation");
      setMaskThreshold(segmentation.threshold ?? .5);
      setMinComponentArea(0);
      setDynamicMask(null);
    }
    else if (explanation) setMode("heatmap");
    else setMode("original");
  }, [result, segmentation, explanation]);

  useEffect(() => {
    if (!segmentation?.probability_png_base64) { setDynamicMask(null); return; }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void renderThresholdMask(segmentation.probability_png_base64!, maskThreshold, minComponentArea)
        .then((next) => { if (!cancelled) setDynamicMask(next); })
        .catch(() => { if (!cancelled) setDynamicMask(null); });
    }, 70);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [segmentation?.probability_png_base64, maskThreshold, minComponentArea]);

  const changePlane = (next: Plane) => {
    setPlane(next);
    if (volume) setSlice(Math.floor(((next === "Axial" ? volume.dims[2] : next === "Coronal" ? volume.dims[1] : volume.dims[0]) - 1) / 2));
  };
  const maskUrl = dynamicMask?.maskUrl ?? (segmentation ? `data:image/png;base64,${segmentation.mask_png_base64}` : null);
  const maskOverlayUrl = dynamicMask?.overlayUrl ?? (segmentation ? `data:image/png;base64,${segmentation.mask_overlay_png_base64 ?? segmentation.mask_png_base64}` : null);
  const heatmapUrl = explanation ? `data:image/png;base64,${explanation.heatmap_png_base64}` : null;
  const showSegmentation = mode === "segmentation" || mode === "compare";
  const showHeatmap = mode === "heatmap";

  return (
    <section className={"viewer-panel panel " + (patientMode ? "patient-viewer-panel" : "")}>
      <div className="panel-heading viewer-heading">
        <div className="heading-icon"><Layers3 size={18} /></div>
        <div className="heading-copy"><div className="eyebrow">{patientMode ? "SCAN REVIEW" : "IMAGE REVIEW"}</div><h2>{patientMode ? "Your MRI image" : "MRI viewer"}</h2><p>{patientMode ? (file ? "The selected image is ready for review." : "Choose an image to get started.") : file ? file.name : "Load a scan to begin review"}</p></div>
        <div className="viewer-actions">
          {file && <button className="icon-button" onClick={onClear} title="Reset to original / clear scan" aria-label="Reset viewer"><RotateCcw size={16} /></button>}
          <button className="icon-button" onClick={() => fileInput.current?.click()} title="Choose scan" aria-label="Choose scan"><FolderOpen size={16} /></button>
        </div>
      </div>
      <input ref={fileInput} type="file" hidden accept={patientMode ? ".png,.jpg,.jpeg,.bmp,.tif,.tiff" : ".png,.jpg,.jpeg,.bmp,.tif,.tiff,.nii,.nii.gz"} onChange={(event) => onPick(event.target.files?.[0])} />

      {!file ? (
        <div className="dropzone" role="button" tabIndex={0} aria-label="Choose or drop an MRI scan" onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.current?.click(); } }} onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); onPick(event.dataTransfer.files[0]); }}>
          <div className="upload-emblem"><Upload size={21} /></div><strong>{patientMode ? "Add a brain MRI image" : "Open a brain MRI scan"}</strong><span>Drop an image here or <u>browse this device</u></span><small>{patientMode ? "JPG · PNG · TIFF · BMP · up to 64 MB" : "JPG · PNG · TIFF · BMP · NIfTI · up to 64 MB"}</small>
        </div>
      ) : (
        <div className={"scan-stage " + (volume ? "volume-stage" : "")}>
          {imageLike && mode === "compare" && source && maskOverlayUrl ? <div className="compare-panes">
            <div className="compare-pane"><span>Original scan</span><img src={source} alt="Original MRI scan for comparison" /></div>
            <div className="compare-pane"><span>Predicted mask overlay</span><div className="compare-picture"><img src={source} alt="MRI scan under predicted tumor mask" /><img src={maskOverlayUrl} alt="Predicted tumor segmentation mask" style={{ opacity }} /></div></div>
          </div> : imageLike && source ? <>
            {mode === "mask" && maskUrl ? <img className="scan-image mask-only" src={maskUrl} alt="Binary predicted tumor mask only" /> : <>
              <img className="scan-image" src={source} alt="Original MRI scan" />
              {showSegmentation && maskOverlayUrl && <img className="image-layer mask-overlay" style={{ opacity }} src={maskOverlayUrl} alt="Predicted tumor mask overlay" />}
              {showHeatmap && heatmapUrl && <img className="image-layer heatmap-overlay" style={{ opacity }} src={heatmapUrl} alt="Grad-CAM classification attribution heatmap" />}
            </>}
          </> : volume ? <canvas ref={canvas} className="volume-canvas" aria-label={plane + " slice preview"} /> : <div className="volume-error">Reading volume…</div>}
          {volume && <div className="volume-hud"><span>{plane.toUpperCase()}</span><span>{String(slice + 1).padStart(3, "0")} / {String(maxSlice + 1).padStart(3, "0")}</span><span>LOCAL VOXEL PREVIEW</span></div>}
          {imageLike && !patientMode && <div className="scan-hud"><span>2D INPUT</span><span>{file.name.split(".").pop()?.toUpperCase()}</span></div>}
          {busy && <div className="viewer-loading"><span className="spinner" />{IS_HOSTED_DEMO ? "Running hosted research models…" : "Running local models…"}</div>}
          {imageLike && result && !segmentation && <div className="viewer-unavailable"><AlertCircle size={15} /> Predicted segmentation mask unavailable</div>}
        </div>
      )}

      {volume && <div className="slice-controls"><div className="plane-switch">{planes.map((item) => <button type="button" key={item} className={plane === item ? "selected" : ""} onClick={() => changePlane(item)}>{item}</button>)}</div><label><span>Slice <b>{slice + 1}</b> <i>/ {maxSlice + 1}</i></span><input type="range" min="0" max={maxSlice} value={Math.min(slice, maxSlice)} onChange={(event) => setSlice(Number(event.target.value))} /></label></div>}

      {imageLike && <div className={"viewer-tools " + (patientMode ? "patient-viewer-tools" : "")}>
        <div className="view-mode-tabs" role="group" aria-label="Scan display mode">
          <button type="button" className={mode === "original" ? "selected" : ""} onClick={() => setMode("original")}>Original</button>
          <button type="button" disabled={!segmentation} className={mode === "segmentation" ? "selected" : ""} onClick={() => setMode("segmentation")}>{patientMode ? "Highlighted area" : "Tumor overlay"}</button>
          <button type="button" disabled={!segmentation} className={mode === "mask" ? "selected" : ""} onClick={() => setMode("mask")}>Mask only</button>
          <button type="button" disabled={!explanation} className={mode === "heatmap" ? "selected" : ""} onClick={() => setMode("heatmap")}>Grad-CAM</button>
          <button type="button" disabled={!segmentation} className={mode === "compare" ? "selected" : ""} onClick={() => setMode("compare")}>Compare</button>
        </div>
        {(mode === "segmentation" || mode === "heatmap" || mode === "compare") && <label className="opacity-control"><span>{mode === "heatmap" ? "Heatmap opacity" : "Overlay opacity"} <b>{Math.round(opacity * 100)}%</b></span><input type="range" min="0" max="1" step=".01" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /></label>}
        {segmentation?.probability_png_base64 && mode !== "original" && mode !== "heatmap" && <div className="advanced-segmentation-controls">
          <div className="advanced-control-heading"><strong>Advanced mask preview</strong><span>{dynamicMask ? `${dynamicMask.componentCount} regions · ${(100 * dynamicMask.positivePixels / Math.max(1, dynamicMask.totalPixels)).toFixed(2)}% image coverage` : "Preparing mask…"}</span></div>
          <label><span>Pixel-score threshold <b>{maskThreshold.toFixed(2)}</b></span><input type="range" min=".10" max=".90" step=".01" value={maskThreshold} onChange={(event) => setMaskThreshold(Number(event.target.value))} /></label>
          <label><span>Remove regions smaller than <b>{minComponentArea} preview px</b></span><input type="range" min="0" max="1000" step="25" value={minComponentArea} onChange={(event) => setMinComponentArea(Number(event.target.value))} /></label>
          <small>These sliders change only the displayed mask. They do not change the model score or confirm a finding.</small>
        </div>}
        <div className="overlay-legend">
          {(mode === "segmentation" || mode === "compare") && <span><i className="legend-mask" /> Predicted tumor mask</span>}
          {mode === "heatmap" && <span><i className="legend-heat" /> Grad-CAM · areas influencing the model score</span>}
          {mode === "mask" && <span><i className="legend-mask" /> Predicted region mask · white is selected</span>}
          {mode === "original" && <span>Showing the unannotated scan</span>}
          {mode === "compare" && <span>Original scan beside the model mask overlay</span>}
        </div>
      </div>}

      {volume && !patientMode && <div className="volume-note"><AlertCircle size={14} /> Volume preview stays in the browser. This project’s trained models support 2D images only.</div>}
      {!patientMode && <div className="viewer-footer"><span><i className="status-dot-local" /> {file ? "LOCAL FILE" : "AWAITING INPUT"}</span><span>{file ? (volume ? volume.dims.slice(0, 3).join(" × ") + " voxels" : `${result?.scan.width ?? "—"} × ${result?.scan.height ?? "—"} px`) : "No file loaded"}</span></div>}
    </section>
  );
}
