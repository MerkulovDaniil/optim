function Div(div)
  -- Check if the div has the "plotly" class
  for _, class in ipairs(div.classes) do
    if class == "plotly" then
      -- Extract HTML file path from the div content
      local htmlFilePath = pandoc.utils.stringify(div.content[1])
      
      -- Read the content of the HTML file
      local fileContent
      local fileHandle = io.open(htmlFilePath, "r")
      if fileHandle then
        fileContent = fileHandle:read("*all")
        fileHandle:close()
      else
        error("Failed to read file: " .. htmlFilePath)
        return
      end
      
      -- Create HTML block element with the file content
      local htmlElement = pandoc.RawBlock("html", fileContent)
      
      -- Return the HTML block element
      return htmlElement
    end
  end
end
