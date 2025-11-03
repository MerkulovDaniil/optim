function Div(div)
  -- Check if the div has the "video" class
  for _, class in ipairs(div.classes) do
    if class == "video" then
      -- get src from first inline text
      local videoPath = pandoc.utils.stringify(div.content[1] or "")
      if videoPath == "" then return nil end

      -- optional poster attr on the div: ::: video {poster="thumb.jpg"} :::
      local poster = div.attributes and div.attributes["poster"] or nil
      local posterAttr = poster and (' poster="' .. poster .. '"') or ""

      local html =
        '<div class="responsive-video">' ..
          '<video autoplay loop muted playsinline webkit-playsinline preload="metadata" class="video' ..
          '"' .. posterAttr .. '>' ..
            '<source src="' .. videoPath .. '" type="video/mp4">' ..
            'Your browser does not support the video tag.' ..
          '</video>' ..
        '</div>'

      return pandoc.RawBlock("html", html)
    end
  end
end
