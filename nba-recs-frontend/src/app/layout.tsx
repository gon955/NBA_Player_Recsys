import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Link unfurlers (LinkedIn, Slack, iMessage, X) refuse to build a preview from
// relative image paths, and a static export has no request to infer the host
// from, so the canonical origin has to be baked in at build time. Override it
// with NEXT_PUBLIC_SITE_URL when deploying anywhere other than the production
// Pages project — a preview deployment, or a custom domain.
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "https://nba-recs.pages.dev";

const title = "NBA Player Recommender";
const description =
  "Era-aware NBA roster recommendations, player archetype clusters, and grounded " +
  "natural-language answers over 1999-2025 player and team data.";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title,
  description,
  applicationName: title,
  // LinkedIn rejects a preview outright when og:title/og:description/og:image
  // are missing, even though it crawled the page successfully.
  openGraph: {
    type: "website",
    url: siteUrl,
    siteName: title,
    title,
    description,
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "NBA Player Recommender — era-aware recommendations, archetype clusters, and grounded Q&A",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/og.png"],
  },
  alternates: { canonical: siteUrl },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
