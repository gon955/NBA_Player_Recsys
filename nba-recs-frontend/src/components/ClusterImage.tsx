type ClusterImageProps = {
  label: string;
  src: string;
};

export default function ClusterImage({ label, src }: ClusterImageProps) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">{label}</p>
      {/* Served off the backend's /static mount, not the Next.js image pipeline. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={label}
        className="w-full rounded-lg border border-white/10 bg-slate-900 object-contain"
      />
    </div>
  );
}
