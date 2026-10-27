export type Volume = {
  dims: number[];
  values: Float32Array;
  range: [number, number];
};

const dtypeBytes: Record<number, number> = {
  2: 1, 4: 2, 8: 4, 16: 4, 64: 8, 256: 1, 512: 2, 768: 4,
};

export async function readNifti(file: File): Promise<Volume> {
  let buffer = await file.arrayBuffer();
  if (file.name.toLowerCase().endsWith(".gz")) {
    if (!("DecompressionStream" in window)) throw new Error("This browser cannot decompress .nii.gz files.");
    const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      total += part.value.byteLength;
      if (total > 512 * 1024 * 1024) {
        await reader.cancel();
        throw new Error("The decompressed volume exceeds the 512 MB preview limit.");
      }
      chunks.push(part.value);
    }
    const inflated = new Uint8Array(total);
    let cursor = 0;
    for (const chunk of chunks) { inflated.set(chunk, cursor); cursor += chunk.byteLength; }
    buffer = inflated.buffer;
  }
  if (buffer.byteLength < 352) throw new Error("This file is too small to be a NIfTI volume.");
  const view = new DataView(buffer);
  const little = view.getInt32(0, true) === 348;
  if (!little && view.getInt32(0, false) !== 348) throw new Error("This is not a supported NIfTI-1 file.");
  const getI16 = (offset: number) => view.getInt16(offset, little);
  const ndim = getI16(40);
  const dims = Array.from({ length: Math.min(4, Math.max(3, ndim)) }, (_, i) => getI16(42 + i * 2));
  if (dims.some((dim) => dim < 1)) throw new Error("The volume has invalid dimensions.");
  const datatype = getI16(70);
  const bytes = dtypeBytes[datatype];
  if (!bytes) throw new Error("This NIfTI voxel type is not supported in the browser preview.");
  const rawOffset = view.getFloat32(108, little);
  if (!Number.isFinite(rawOffset)) throw new Error("The NIfTI voxel-data offset is invalid.");
  const offset = Math.max(352, Math.floor(rawOffset));
  const n = dims.slice(0, 3).reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(n) || n > 40000000 || offset + n * bytes > buffer.byteLength) throw new Error("The NIfTI voxel data appears truncated or exceeds the 40 million voxel preview limit.");
  const slope = view.getFloat32(112, little) || 1;
  const intercept = view.getFloat32(116, little) || 0;
  const read = (p: number) => {
    switch (datatype) {
      case 2: return view.getUint8(p);
      case 4: return view.getInt16(p, little);
      case 8: return view.getInt32(p, little);
      case 16: return view.getFloat32(p, little);
      case 64: return view.getFloat64(p, little);
      case 256: return view.getInt8(p);
      case 512: return view.getUint16(p, little);
      case 768: return view.getUint32(p, little);
      default: return 0;
    }
  };
  const values = new Float32Array(n);
  let min = Infinity; let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const value = read(offset + i * bytes) * slope + intercept;
    values[i] = value;
    if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  }
  return { dims, values, range: [min, max] };
}

export function drawSlice(canvas: HTMLCanvasElement, volume: Volume, plane: "Axial" | "Coronal" | "Sagittal", slice: number) {
  const [x, y, z] = volume.dims;
  const width = plane === "Sagittal" ? y : x;
  const height = plane === "Axial" ? y : z;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  canvas.width = width; canvas.height = height;
  const image = ctx.createImageData(width, height);
  const [min, max] = volume.range;
  const span = max - min || 1;
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let ix = px; let iy = 0; let iz = 0;
      if (plane === "Axial") { iy = y - 1 - py; iz = slice; }
      else if (plane === "Coronal") { iy = slice; iz = z - 1 - py; }
      else { ix = slice; iy = px; iz = z - 1 - py; }
      const value = volume.values[ix + x * (iy + y * iz)];
      const gray = Math.max(0, Math.min(255, Math.round((value - min) / span * 255)));
      const at = 4 * (px + width * py);
      image.data[at] = gray; image.data[at + 1] = gray; image.data[at + 2] = gray; image.data[at + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}
