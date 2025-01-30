function Div(div)
  -- Check if the div has the "video" class
  for _, class in ipairs(div.classes) do
    if class == "video" then
      -- Extract video file path from the div content
      local videoPath = pandoc.utils.stringify(div.content[1])
      
      -- Create video element with required attributes for Chrome
      local videoElement = pandoc.RawBlock("html", 
      '<div class="responsive-video">' ..
      '<video autoplay loop muted playsinline class="video">' ..
      '<source src="' .. videoPath .. '" type="video/mp4">' ..
      'Your browser does not support the video tag.' ..
      '</video></div>')
      
      -- Return the video element
      return videoElement
    end
  end
end
