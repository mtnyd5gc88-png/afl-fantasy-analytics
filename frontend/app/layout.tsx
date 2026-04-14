import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AFL Fantasy Analytics",
  description: "Projections, risk, and trade optimization",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
