"use client";

import React, { useRef } from 'react';
import classNames from 'classnames';
import { Popover } from "flowbite-react";
import { useInView } from "framer-motion";

interface DriverCardProps {
    championshipLevel?: string;
    className?: string;
    driver: { code: string, [key: string]: any };
    driverColor?: string;
    stint?: any[];
    fastestLap?: any;
    status?: string;
    startPosition?: number;
    endPosition?: number;
    isActive?: boolean;
    layoutSmall?: boolean;
    time?: string;
    year?: number;
    hasHover?: boolean;
    index: number;
    mobileSmall?: boolean;
    isRace?: boolean;
    darkBG?: boolean;
}

export const DriverCard: React.FC<DriverCardProps> = (props) => {
    const { championshipLevel, className, driver, driverColor, stint, fastestLap, status, startPosition = 0, endPosition = 0, isActive, layoutSmall, time, year = 2026, hasHover, index, mobileSmall, isRace, darkBG} = props;

    const ref = useRef(null);
    const isInView = useInView(ref, { once: true });

    const getTireCompound = (driverCode: string, lap: number) => {
        const driverStint = stint?.find(item => item.acronym === driverCode);
        if (driverStint && driverStint.tires) {
            for (const tire of driverStint.tires){
                if(lap <= tire.lap_end){
                    return tire.compound;
                }
            }
        }
        return '?';
    } 

    const positionMovement = () => {
        if ( startPosition !== endPosition ) {
            return (
                <Popover
                    aria-labelledby="default-popover"
                    className="bg-glow border-neutral-400 border-[.1rem] p-4 bg-neutral-950 rounded-md z-[10]"
                    trigger="hover"
                    placement="top"
                    arrow={false}
                    content={
                        <div className="p-4">
                            <div>
                                <span className="text-sm mr-4">Started</span>
                                <span className="font-display">P{startPosition}</span></div>
                            <div>
                                <span className="text-sm mr-4">Ended</span>
                                <span className="font-display">P{endPosition}</span>
                            </div>
                        </div>
                    }
                >
                    <div className={classNames("text-xs font-bold", startPosition > endPosition ? "text-emerald-500" : "text-rose-500")}>
                        {startPosition > endPosition ? "▲" : "▼"} Math.abs({startPosition - endPosition})
                    </div>
                </Popover>
            )
        }
        return null;
    }

    const driverImage = (
        <img 
            alt="Driver" 
            src={championshipLevel && championshipLevel !== 'f1' ? 
                `/images/${year}/${championshipLevel}/${driver.code}.png`
                : `/images/${year}/drivers/${driver.code}.png`
            }
            ref={ref}
            className={classNames('absolute block inset-x-0 bottom-[-10px] h-[130px] w-auto max-w-none object-cover mix-blend-screen scale-110', championshipLevel === 'F2' ? 'left-[46px] sm:left-46 rounded-r-md' : 'left-[35px] sm:left-46')}
            style={{
                opacity: isInView ? 1 : 0,
                transition: `all 1s cubic-bezier(0.17, 0.55, 0.55, 1) .${index}s`
            }}
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
        />
    )

    const isFastestLapDriver = String(fastestLap?.rank) === "1";
    const fastestLapTime = fastestLap?.Time?.time || fastestLap?.time?.time || fastestLap?.Time || fastestLap?.time || "";
    const fastestLapAverageSpeed = fastestLap?.AverageSpeed || fastestLap?.averageSpeed;
    
    return (
        <div 
            className={classNames(
                className, 
                'driver-card flex items-stretch bg-glow relative overflow-hidden h-[120px] shadow-sm',
                { 
                    'driver-card--canvas': mobileSmall,
                    'hidden': status === "cancelled",
                    'bg-neutral-800' : darkBG,
                },
                isActive ? "bg-glow--active" : hasHover ? "bg-glow--hover" : "",
                mobileSmall ? "rounded-sm mb-4" : "rounded-t-md"
            )}
            style={{borderColor: isActive ? `#${driverColor}` : undefined}}
        >
            <div className={classNames("flex items-center justify-between w-full h-full", { "max-md:hidden": mobileSmall, "hidden": !layoutSmall})}>
                <div className="flex items-center font-display leading-none text-sm h-full">
                    <p className={classNames("w-48 bg-neutral-600 h-full py-4 flex items-center justify-center text-center rounded-l-md")}>P{isRace ? endPosition : index + 1}</p>
                    <span className="pl-16 mr-4">{driver.code}</span>
                </div>
                <p className=" text-xs pr-8">{time}</p>
            </div>
            <div className={classNames('flex items-center w-full h-full relative z-10', { "max-md:hidden": mobileSmall, "hidden": layoutSmall})}>
                <div className={classNames("driver-card-position w-28 flex-shrink-0 text-center h-full text-[28px] font-display bg-neutral-700/80 rounded-l-md flex items-center justify-center backdrop-blur-md relative z-20 shadow-md")}>
                    P{isRace ? endPosition : index + 1}
                </div>
                {driverImage}
                <div className="grow py-4 px-12 text-right relative z-20 flex flex-col justify-center h-full bg-gradient-to-r from-transparent via-neutral-900/50 to-neutral-900">
                    <span className="heading-4 mb-2 pl-32 text-4xl font-black italic uppercase tracking-wider text-white drop-shadow-md">{driver.code}</span>
                    <div className="divider-glow w-full opacity-50 mb-2" /> 
                    <p className={classNames("font-bold tracking-widest text-[#00d2be] uppercase text-sm")}>{time}</p>
                </div>
            </div>
            
            <div className="fastest-lap-popover popover-wrapper flex flex-col items-center absolute -right-8">
                {isRace && positionMovement()}
            </div>
        </div>
    );
};
