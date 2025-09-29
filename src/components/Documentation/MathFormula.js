import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const MathFormula = ({ formula, inline = false }) => {
  try {
    if (inline) {
      return (
        <span className="inline-math">
          <InlineMath math={formula} />
        </span>
      );
    } else {
      return (
        <div className="block-math my-6 p-4 bg-gray-50 border border-gray-300 overflow-x-auto overflow-y-hidden max-w-full">
          <BlockMath math={formula} />
        </div>
      );
    }
  } catch (error) {
    console.error('KaTeX rendering error:', error);
    return (
      <span className="math-error text-red-700 bg-red-100 px-2 py-1 text-sm border border-red-400">
        Math Error: {formula}
      </span>
    );
  }
};

export default MathFormula;