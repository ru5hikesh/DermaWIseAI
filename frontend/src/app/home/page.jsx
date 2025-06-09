'use client';

import { useState } from 'react';
import Footer from "@/components/ui/Footer";
import LeftBox from "@/components/ui/LeftBox";
import Navbar from "@/components/ui/Navbar";
import RightBox from "@/components/ui/RightBox";

export default function Home() {
  const message = "Powered by DermaWiseAI";
  const [diagnosisData, setDiagnosisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="min-h-screen min-w-full bg-[#0a0a0a] pt-[70px]">
      <Navbar />
      <div className="max-w-[1400px] mx-auto my-8 pb-0 px-8 flex flex-col gap-8">
        <div className="flex gap-8 mb-8 justify-between items-start flex-col md:flex-row">
          <LeftBox 
            setDiagnosisData={setDiagnosisData} 
            setIsLoading={setIsLoading} 
          />
          <RightBox 
            diagnosisData={diagnosisData} 
            isLoading={isLoading} 
          />
        </div>
        <Footer message={message} />
      </div>
    </div>
  );
}