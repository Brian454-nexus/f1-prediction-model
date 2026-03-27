import type { Metadata } from "next";
import { Lato, Bungee } from "next/font/google";
import "./globals.css";

const lato = Lato({ 
  subsets: ["latin"], 
  weight: ["100", "300", "400", "700", "900"],
  variable: '--font-lato'
});

const bungee = Bungee({ 
  subsets: ["latin"], 
  weight: ["400"],
  variable: '--font-bungee' 
});

export const metadata: Metadata = {
  title: "F1 APEX - Pit Wall",
  description: "F1 APEX 2026 Prediction Engine Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${lato.className} ${lato.variable} ${bungee.variable} bg-black text-white`}>
        {children}
      </body>
    </html>
  );
}
