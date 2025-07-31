// // tailwind.config.js

// const {
//   default: flattenColorPalette,
// } = require("tailwindcss/lib/util/flattenColorPalette");

// /** @type {import('tailwindcss').Config} */
// module.exports = {
//   content: [
//     "./src/**/*.{js,jsx,ts,tsx}", // Adjust this to your project structure
//   ],
//   darkMode: "class",
//   theme: {
//     extend: {
//       colors: {
//         // As per the TRD
//         'ubs-black': '#000000',
//         'ubs-red': '#e60000',
//         'ubs-white': '#ffffff',
//         'ubs-red-web': '#da0000',
//         'ubs-bordeaux1': '#bd000c',
//         'ubs-bordeaux50': '#b03974',
//         'ubs-sand': '#cfbd9b',
//         'ubs-caramel': '#cfbd9b',
//         'ubs-ginger': '#e05bd0',
//         'ubs-chocolate': '#4d3c2f',
//         'ubs-clay': '#7b6b59',
//         'ubs-mouse': '#beb29e',
//         'ubs-curry': '#e5b01c',
//         'ubs-amber-web': '#f2c551',
//         'ubs-warm5': '#5b5e5d',
//         'ubs-honey': '#edc860',
//         'ubs-straw': '#f2d88e',
//         'ubs-chestnut-web': '#ba0000',
//         'ubs-chestnut': '#a43725',
//         'ubs-terracotta': '#c07156',
//         'ubs-cinnamon': '#e6b644',
//       },
//       animation: {
//           spotlight: "spotlight 2s ease .75s 1 forwards",
//       },
//       keyframes: {
//           spotlight: {
//               "0%": {
//                   opacity: 0,
//                   transform: "translate(-72%, -62%) scale(0.5)",
//               },
//               "100%": {
//                   opacity: 1,
//                   transform: "translate(-50%,-40%) scale(1)",
//               },
//           },
//       },
//     },
//   },
//   plugins: [addVariablesForColors],
// };

// function addVariablesForColors({ addBase, theme }) {
//   let allColors = flattenColorPalette(theme("colors"));
//   let newVars = Object.fromEntries(
//     Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
//   );

//   addBase({
//     ":root": newVars,
//   });
// }


// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ubs-black': '#000000',
        'ubs-red': '#e60000',
        'ubs-white': '#ffffff',
        'ubs-red-web': '#da0000',
        'ubs-bordeaux1': '#bd000c',
        'ubs-bordeaux50': '#b03974',
        'ubs-sand': '#cfbd9b',
        'ubs-caramel': '#cfbd9b',
        'ubs-ginger': '#e05bd0',
        'ubs-chocolate': '#4d3c2f',
        'ubs-clay': '#7b6b59',
        'ubs-mouse': '#beb29e',
        'ubs-curry': '#e5b01c',
        'ubs-amber-web': '#f2c551',
        'ubs-warm5': '#5b5e5d',
        'ubs-honey': '#edc860',
        'ubs-straw': '#f2d88e',
        'ubs-chestnut-web': '#ba0000',
        'ubs-chestnut': '#a43725',
        'ubs-terracotta': '#c07156',
        'ubs-cinnamon': '#e6b644',
      }
    }
  }
}