import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

const Cursor = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const updateMousePosition = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    const handleMouseOver = (e) => {
      if (
        e.target.tagName === 'BUTTON' ||
        e.target.tagName === 'A' ||
        e.target.tagName === 'INPUT' ||
        e.target.closest('button') ||
        e.target.closest('a') ||
        e.target.classList.contains('cursor-pointer')
      ) {
        setIsHovering(true);
      } else {
        setIsHovering(false);
      }
    };

    window.addEventListener('mousemove', updateMousePosition);
    window.addEventListener('mouseover', handleMouseOver);

    return () => {
      window.removeEventListener('mousemove', updateMousePosition);
      window.removeEventListener('mouseover', handleMouseOver);
    };
  }, []);

  return (
    <>
      {/* Crosshair Center */}
      <motion.div
        className="fixed top-0 left-0 w-2 h-2 bg-cyan-400 rounded-full pointer-events-none z-[9999] mix-blend-exclusion"
        animate={{
          x: mousePosition.x - 4,
          y: mousePosition.y - 4,
          scale: isHovering ? 0.5 : 1,
        }}
        transition={{ type: 'spring', stiffness: 1000, damping: 50 }}
      />

      {/* Crosshair Ring */}
      <motion.div
        className={clsx(
          "fixed top-0 left-0 w-12 h-12 border border-cyan-400/50 rounded-full pointer-events-none z-[9998] mix-blend-exclusion flex items-center justify-center",
          isHovering ? "bg-cyan-400/10" : "bg-transparent"
        )}
        animate={{
          x: mousePosition.x - 24,
          y: mousePosition.y - 24,
          scale: isHovering ? 1.5 : 1,
          rotate: isHovering ? 90 : 0,
        }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      >
        {/* Crosshair Lines */}
        <div className="absolute w-full h-[1px] bg-cyan-400/30" />
        <div className="absolute h-full w-[1px] bg-cyan-400/30" />
      </motion.div>
    </>
  );
};

export default Cursor;
