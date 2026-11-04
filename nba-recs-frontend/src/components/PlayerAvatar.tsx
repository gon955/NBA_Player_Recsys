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
    <div className="flex h-16 w-14 shrink-0 items-center justify-center overflow-hidden bg-ink/10">
      {photoUrl && !failed ? (
        // Headshots come from basketball-reference, which is not configured as a
        // next/image remote pattern; a plain img keeps the fallback logic simple.
        // Multiply lays the white photo background into the paper, like newsprint.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={photoUrl}
          alt=""
          loading="lazy"
          className="h-full w-full object-cover object-top grayscale mix-blend-multiply"
          onError={() => setFailed(true)}
        />
      ) : (
        <span aria-hidden className="text-sm font-bold text-muted">
          {initials}
        </span>
      )}
    </div>
  );
}
