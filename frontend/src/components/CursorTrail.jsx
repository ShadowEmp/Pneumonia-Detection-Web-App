import React, { useEffect, useRef } from 'react';

const CursorTrail = () => {
    const canvasRef = useRef(null);
    const particles = useRef([]);
    const mouse = useRef({ x: 0, y: 0 });
    const lastMouse = useRef({ x: 0, y: 0 }); // Track previous position for interpolation

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        const handleMouseMove = (e) => {
            mouse.current = { x: e.clientX, y: e.clientY };

            // Initialize lastMouse on first move if needed
            if (lastMouse.current.x === 0 && lastMouse.current.y === 0) {
                lastMouse.current = { ...mouse.current };
            }
        };

        window.addEventListener('mousemove', handleMouseMove);

        const createParticle = (x, y) => {
            return {
                x,
                y,
                size: Math.random() * 2.5 + 0.5, // Slightly smaller for smoother look
                speedX: Math.random() * 1 - 0.5, // Reduced spread
                speedY: Math.random() * 1 - 0.5,
                life: 1,
                decay: Math.random() * 0.015 + 0.005, // Slower decay
                color: `hsl(${Math.random() * 40 + 180}, 100%, 50%)` // Cyan/Blue
            };
        };

        const lerp = (start, end, t) => {
            return start * (1 - t) + end * t;
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Interpolation logic: Emit particles along the path from lastMouse to currentMouse
            const dx = mouse.current.x - lastMouse.current.x;
            const dy = mouse.current.y - lastMouse.current.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Number of steps based on distance (more steps = smoother line)
            const steps = Math.max(Math.abs(dx), Math.abs(dy)) / 2;

            if (steps > 0) {
                for (let i = 0; i < steps; i++) {
                    const t = i / steps;
                    const x = lerp(lastMouse.current.x, mouse.current.x, t);
                    const y = lerp(lastMouse.current.y, mouse.current.y, t);

                    // Randomly emit particles along the path
                    if (Math.random() < 0.3) { // Density control
                        particles.current.push(createParticle(x, y));
                    }
                }
            }

            // Update lastMouse to current for next frame
            lastMouse.current = { ...mouse.current };

            // Render particles
            particles.current.forEach((particle, index) => {
                particle.x += particle.speedX;
                particle.y += particle.speedY;
                particle.life -= particle.decay;
                particle.size -= 0.02;

                if (particle.life <= 0 || particle.size <= 0) {
                    particles.current.splice(index, 1);
                } else {
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                    ctx.fillStyle = particle.color.replace(')', `, ${particle.life})`);
                    ctx.fill();

                    ctx.shadowBlur = 5;
                    ctx.shadowColor = particle.color;
                }
            });

            // Connect particles (optimized distance check)
            ctx.lineWidth = 0.3;
            for (let i = 0; i < particles.current.length; i++) {
                // Only check a subset or neighbors to save performance if needed, 
                // but for < 100 particles O(N^2) is fine.
                // Optimization: Only connect to particles created recently (end of array)
                // or just limit the loop.

                for (let j = i + 1; j < particles.current.length; j++) {
                    const dx = particles.current[i].x - particles.current[j].x;
                    const dy = particles.current[i].y - particles.current[j].y;

                    // Quick check before sqrt
                    if (Math.abs(dx) > 50 || Math.abs(dy) > 50) continue;

                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 50) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(6, 182, 212, ${0.15 * particles.current[i].life})`;
                        ctx.moveTo(particles.current[i].x, particles.current[i].y);
                        ctx.lineTo(particles.current[j].x, particles.current[j].y);
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            window.removeEventListener('mousemove', handleMouseMove);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-[9997]"
        />
    );
};

export default CursorTrail;
