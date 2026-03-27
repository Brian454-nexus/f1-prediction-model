"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { DriverCard } from "@/components/DriverCard";
import { ConstructorCar } from "@/components/ConstructorCar";
import { motion, AnimatePresence } from "framer-motion";

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    const errorData = await res.json();
    throw new Error(errorData.detail || "Error loading prediction data");
  }
  return res.json();
};

export default function Home() {
  const { data, error, isLoading } = useSWR("http://localhost:8000/prediction/2", fetcher);

  if (isLoading) return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-plum-500"></div>
    </div>
  );

  if (error) return (
    <div className="flex items-center justify-center min-h-screen bg-black text-rose-500 font-display text-2xl text-center px-4">
      {error.message} <br/><br/>
      💡 Hint: You need to trigger the model inference by sending a POST request to /predict/2 first!
    </div>
  );

  if (!data) return null;

  const predictions = data.predictions || [];
  const top3 = predictions.slice(0, 3);
  const rest = predictions.slice(3, 20);

  // Helper mappings for visual assets based on team (using realistic 2026/F1nsight asset mocks)
  const getTeamColor = (team: string) => {
    switch (team) {
      case 'Red Bull': return '#0600ef';
      case 'Ferrari': return '#dc0000';
      case 'Mercedes': return '#00d2be';
      case 'Aston Martin': return '#006f62';
      case 'McLaren': return '#ff8700';
      default: return '#ffffff';
    }
  };

  const getCarImage = (team: string) => {
    if (team === 'Racing Bulls') return 'rb';
    return team.toLowerCase().replace(' ', '_');
  };

  return (
    <main className="min-h-screen bg-black bg-gradient-f1a pb-32">
        <div className="global-container page-container-centered pt-16">
            <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="text-center mb-16"
            >
                <h1 className="text-6xl md:text-8xl font-display uppercase tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-plum-300 to-white">
                    {data.circuit_name} GP '26
                </h1>
                <p className="text-xl md:text-2xl text-neutral-400 mt-4 tracking-widest uppercase font-bold">
                    F1 APEX Prediction • Model: <span className="text-plum-300">{data.model_branch}</span>
                </p>
                {data.weather?.weather_status === "Rain" && (
                    <div className="mt-4 inline-block px-4 py-1 rounded bg-blue-900 text-blue-200 border border-blue-500 text-sm font-bold uppercase tracking-widest">
                        Wet Weather Simulation Mode Active
                    </div>
                )}
            </motion.div>

            {/* Podium Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-24 items-end">
                {top3.map((pred: any, i: number) => {
                   return (
                    <motion.div 
                        key={pred.driver}
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: i * 0.2 }}
                        className={i === 0 ? "order-1 md:order-2 md:-translate-y-12" : i === 1 ? "order-2 md:order-1" : "order-3 md:order-3"}
                    >
                        <ConstructorCar
                            championshipLevel="f1"
                            color={getTeamColor(pred.team)}
                            points={`${(pred.win_probability * 100).toFixed(1)}%`}
                            image={getCarImage(pred.team)}
                            name={pred.team}
                            year={2026}
                            drivers={[pred.driver.substring(0, 3).toUpperCase(), '', '', '']}
                            index={i}
                        />
                        <div className="text-center mt-4">
                            <span className="text-xs text-neutral-400 font-bold tracking-widest uppercase">Win Probability</span>
                        </div>
                    </motion.div>
                   )
                })}
            </div>

            <div className="divider-glow w-full mb-16" />

            <h2 className="text-3xl font-display mb-8 text-neutral-200 uppercase tracking-widest">Rest of the Grid</h2>

            {/* Rest of the Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-10 px-2 sm:px-4">
                <AnimatePresence>
                    {rest.map((pred: any, i: number) => {
                        // Find a non-ESS explanation to avoid repetition, or fallback
                        const validExplanation = Array.isArray(pred.explanation) 
                            ? pred.explanation.find((e: any) => e.factor !== 'Energy Strategy (ESS)') || pred.explanation[1] || pred.explanation[0]
                            : { description: "Standard race pace predicted" };
                        
                        return (
                        <motion.div
                            key={pred.driver}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4, delay: i * 0.05 }}
                            className="flex flex-col"
                        >
                            <DriverCard
                                className="w-full h-[120px]"
                                championshipLevel="f1"
                                driver={{ code: pred.driver.substring(0, 3).toUpperCase(), name: pred.driver }}
                                driverColor={getTeamColor(pred.team).replace('#', '')}
                                isRace={true}
                                index={i + 3}
                                endPosition={i + 4}
                                startPosition={Math.round(pred.predicted_position + (Math.random() * 4 - 2))}
                                time={`Top 10: ${(pred.top10_probability * 100).toFixed(1)}%`}
                                hasHover={true}
                                mobileSmall={false}
                                darkBG={true}
                            />
                            {/* SHAP Explanation Snippet */}
                            <div className="px-5 py-4 bg-neutral-900/80 border border-neutral-800 rounded-b-md text-sm text-neutral-300 mt-[-4px] backdrop-blur-sm relative z-10 shadow-lg leading-relaxed flex-grow">
                                <span className="text-plum-400 font-bold uppercase tracking-widest text-xs mr-2">SHAP insight:</span> 
                                {validExplanation?.description || "Standard race pace predicted"}
                            </div>
                        </motion.div>
                    )})}
                </AnimatePresence>
            </div>
        </div>
    </main>
  );
}
