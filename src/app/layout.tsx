import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { PyodideProvider } from "@/contexts/pyodide-context";
import { ProgressProvider } from "@/contexts/progress-context";
import { ThemeProvider } from "@/providers/theme-provider";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppSidebar } from "@/components/app-sidebar";
import { GlobalCommandMenu } from "@/components/global-command-menu";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Scikit-Learn Practice",
  description: "Practice scikit-learn in your browser. No installation needed.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <ProgressProvider>
            <PyodideProvider>
              <TooltipProvider>
                <SidebarProvider>
                  <AppSidebar />
                  <SidebarInset>
                    <main className="min-h-screen bg-background">{children}</main>
                  </SidebarInset>
                  <GlobalCommandMenu />
                </SidebarProvider>
              </TooltipProvider>
            </PyodideProvider>
          </ProgressProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
