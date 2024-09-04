module Jekyll
    class JupyterConverter < Converter
      safe true
      priority :low
  
      def matches(ext)
        ext =~ /^\.ipynb$/i
      end
  
      def output_ext(ext)
        ".html"
      end
  
      def convert(content)
        Jekyll.logger.info "JupyterConverter:", "Converting content for Jupyter Notebook"
        begin
          # Write the notebook content to a temporary file
          require 'tempfile'
          notebook_file = Tempfile.new(['notebook', '.ipynb'])
          notebook_file.write(content)
          notebook_file.close
  
          # Use nbconvert to convert the notebook to HTML
          output = `jupyter nbconvert --to html --stdout #{notebook_file.path}`
          
          # Clean up the temporary file
          notebook_file.unlink
          
          # Return the HTML content
          return output
        rescue => e
          Jekyll.logger.error "JupyterConverter:", "Error converting notebook: #{e}"
          raise e
        end
      end
    end
  end