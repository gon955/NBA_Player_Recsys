"use client";

import { useState } from "react";

type PlayerAvatarProps = {
  name: string;
  photoUrl?: string | null;
};

export default function PlayerAvatar({ name, photoUrl }: PlayerAvatarProps) {
  const [failed, setFailed] = useState(false);
  const initials = name
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <div className="h-16 w-16 shrink-0 flex items-center justify-center rounded-2xl bg-slate-800 ring-1 ring-white/10 overflow-hidden">
      {photoUrl && !failed ? (
        // Headshots come from basketball-reference, which is not configured as a
        // next/image remote pattern; a plain img keeps the fallback logic simple.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={photoUrl}
          alt={`${name} headshot`}
          className="h-full w-full object-contain bg-white"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="text-sm font-semibold text-slate-400">{initials}</span>
      )}
    </div>
  );
}
