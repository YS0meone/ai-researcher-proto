export function CorvusSVG({
  className,
  width,
  height,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="corvusGrad" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#222222" />
          <stop offset="100%" stopColor="#aaaaaa" />
        </linearGradient>
      </defs>
      <path
        d="M4,58 C15,58 25,50 35,35 L58,4 L60,20 C50,35 38,48 20,52 L4,58 Z"
        fill="url(#corvusGrad)"
      />
    </svg>
  );
}
