import type { Metadata } from "next";
import Link from "next/link";
import SiteNav from "@/components/site-nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "LesionShiftAI",
  description:
    "Cross-dataset skin lesion classification benchmark spanning baseline CNNs, ensembles, and Vision Transformers."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="page-bg" />
        <header className="site-header">
          <div className="site-shell">
            <div className="brand-row">
              <Link href="/" className="brand-link">
                LesionShiftAI
              </Link>
              <p className="brand-tag">
                Cross-dataset skin lesion classification under dataset shift
              </p>
            </div>
            <SiteNav />
          </div>
        </header>
        <main className="site-shell site-main">{children}</main>
        <footer className="site-footer">
          <div className="site-shell">
            <p>
              Research benchmark only. Not a clinical diagnostic tool. © 2026
              Jeffrey Hoelzel Jr.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}

