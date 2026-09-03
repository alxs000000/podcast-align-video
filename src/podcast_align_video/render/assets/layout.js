(() => {
  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

  async function fontsReady() {
    await document.fonts.ready;
    await document.fonts.load('590 76px "Geist"', "Ag");
    if (!document.fonts.check('590 76px "Geist"', "Ag")) {
      throw new Error("Bundled Geist font did not load");
    }
  }

  function fitSubtitle() {
    const frame = document.querySelector(".subtitle-frame");
    const line = document.querySelector(".subtitle-line");
    const maximum = window.innerWidth <= 820
      ? 52
      : Math.min(76, Math.max(56, window.innerWidth * 0.046));
    let low = 16;
    let high = maximum;
    while (high - low > 0.5) {
      const size = (low + high) / 2;
      line.style.fontSize = `${size}px`;
      const fits = line.scrollHeight <= frame.clientHeight && line.scrollWidth <= frame.clientWidth;
      if (fits) low = size;
      else high = size;
    }
    line.style.fontSize = `${low}px`;
  }

  window.podcastAlignVideoMeasure = async (words) => {
    if (!Array.isArray(words) || words.some((word) => typeof word !== "string")) {
      throw new Error("layout input must be an array of strings");
    }
    await fontsReady();
    const line = document.getElementById("subtitle-line");
    line.replaceChildren();
    for (const text of words) {
      const element = document.createElement("button");
      element.className = "subtitle-word";
      element.textContent = text;
      line.appendChild(element);
    }
    fitSubtitle();
    await nextFrame();
    await nextFrame();
    const measured = [...document.querySelectorAll(".subtitle-word")].map((element) => {
      const rect = element.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(element);
      const textRect = range.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        textX: textRect.x,
        textY: textRect.y,
        textWidth: textRect.width,
        textHeight: textRect.height,
        fontSize: parseFloat(style.fontSize),
        fontWeight: parseInt(style.fontWeight, 10),
        text: element.textContent || "",
      };
    });
    if (measured.length !== words.length) {
      throw new Error(`DOM word count mismatch: expected ${words.length}, got ${measured.length}`);
    }
    return measured;
  };
})();
