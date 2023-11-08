function Div(div)
  -- Extract the callout type from the class name
  local calloutType = nil
  for _, class in ipairs(div.classes) do
    if class:match("^callout%-") then
      calloutType = class:match("^callout%-(.*)")
      break
    end
  end

  if calloutType then
    -- Default title and collapse setting
    local title
    local collapse = nil

    -- Specific handling for "answer" and "hint" callouts
    if calloutType == "answer" or calloutType == "hint" or calloutType == "proof" or calloutType == "solution" then
      title = calloutType:sub(1,1):upper() .. calloutType:sub(2)
      collapse = true
    else
      -- Replace hyphens with spaces and capitalize each word for the default title
      title = calloutType:gsub("(%a)([%w_]*)", function(first, rest)
        return first:upper() .. rest:lower()
      end):gsub("%-", " ")
    end

    -- Use first element of div as title if this is a header
    if div.content[1] ~= nil and div.content[1].t == "Header" then
      title = pandoc.utils.stringify(div.content[1])
      div.content:remove(1)
    end

    -- Replace hyphens with underscores for the callout type
    calloutType = calloutType:gsub("%-", "_")

    -- Return a callout instead of the Div
    return quarto.Callout({
      type = calloutType,
      content = { pandoc.Div(div) },
      title = title,
      collapse = collapse
    })
  end
end